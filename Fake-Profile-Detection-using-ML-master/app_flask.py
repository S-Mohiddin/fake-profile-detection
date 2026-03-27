from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Any, Optional

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_BASE_DIR, ".env"))
except ImportError:
    pass

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import requests

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-insecure-change-me")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB file upload

# RapidAPI (Live prediction). Prefer a `.env` file (see `.env.example`) or set env vars — never commit keys.
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "").strip()
RAPIDAPI_HOST = os.environ.get(
    "RAPIDAPI_HOST",
    "instagram-scraper-stable-api.p.rapidapi.com",
).strip()
RAPIDAPI_PROFILE_URL = os.environ.get(
    "RAPIDAPI_PROFILE_URL",
    "https://instagram-scraper-stable-api.p.rapidapi.com/ig_get_fb_profile_v3.php",
).strip()

FEATURES = [
    "statuses_count",
    "followers_count",
    "friends_count",
    "favourites_count",
    "listed_count",
    "lang_code",
]

CSV_FEATURE_ALIASES: dict[str, str] = {
    # canonical -> common alternatives
    "statuses_count": "statuses",
    "followers_count": "followers",
    "friends_count": "friends",
    "favourites_count": "favourites",
    "listed_count": "listed",
    "lang_code": "language_code",
}

MODEL_PATHS = {
    "rf": "rf_model.pkl",
    "svm": "svm_model.pkl",
    "nn": "nn_model.pkl"
}

@dataclass(frozen=True)
class ModelArtifact:
    model: Any
    label_encoder: Optional[Any] = None
    feature_names: tuple[str, ...] = tuple(FEATURES)

# Global dictionary to hold the loaded models
ARTIFACTS = {
    "rf": None,
    "svm": None,
    "nn": None
}

def load_artifact(model_path: str) -> Optional[ModelArtifact]:
    if not os.path.exists(model_path):
        return None
    raw = joblib.load(model_path)
    if hasattr(raw, "predict"):
        return ModelArtifact(model=raw, label_encoder=None)
    if isinstance(raw, dict) and "model" in raw:
        return ModelArtifact(
            model=raw["model"],
            label_encoder=raw.get("label_encoder"),
            feature_names=tuple(raw.get("feature_names", FEATURES)),
        )
    return None

def _predict(artifact: ModelArtifact, X: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
    pred = artifact.model.predict(X)
    proba = None
    if hasattr(artifact.model, "predict_proba"):
        try:
            proba = artifact.model.predict_proba(X)
        except Exception:
            proba = None
    return pred, proba

def _label(pred: int) -> str:
    return "Genuine" if int(pred) == 1 else "Fake"

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lower = {c: c.strip().lower() for c in df.columns}
    df.rename(columns=lower, inplace=True)
    reverse_aliases = {v: k for k, v in CSV_FEATURE_ALIASES.items()}
    rename_map: dict[str, str] = {}
    for c in df.columns:
        if c in FEATURES:
            continue
        if c in reverse_aliases:
            rename_map[c] = reverse_aliases[c]
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    return df

def _coerce_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURES:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, dict) and "count" in value:
            return _safe_int(value["count"], default)
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _unwrap_profile_payload(raw: Any, depth: int = 0) -> Optional[dict]:
    """
    RapidAPI responses are often a JSON list, or wrapped in { data: ... }, { user: ... }, etc.
    """
    if raw is None or depth > 8:
        return None
    if isinstance(raw, list):
        for item in raw:
            d = _unwrap_profile_payload(item, depth + 1)
            if isinstance(d, dict) and d:
                return d
        return None
    if not isinstance(raw, dict):
        return None
    # Direct profile-like object
    if any(
        k in raw
        for k in (
            "media_count",
            "follower_count",
            "following_count",
            "username",
            "pk",
            "edge_followed_by",
            "edge_follow",
        )
    ):
        return raw
    for key in ("data", "user", "result", "graphql", "user_data", "graphql_user"):
        inner = raw.get(key)
        if inner is not None:
            d = _unwrap_profile_payload(inner, depth + 1)
            if d is not None:
                return d
    return raw


def _extract_social_counts(d: dict) -> tuple[int, int, int]:
    """Map varied API field names to posts / followers / following."""
    posts = _safe_int(d.get("media_count"))
    followers = _safe_int(d.get("follower_count"))
    following = _safe_int(d.get("following_count"))

    if posts == 0:
        posts = _safe_int(d.get("mediacount"))
    if followers == 0:
        followers = _safe_int(d.get("followers_count"))
    if following == 0:
        following = _safe_int(d.get("friends_count"))

    posts = max(posts, _safe_int(d.get("posts_count")), _safe_int(d.get("posts")))
    followers = max(followers, _safe_int(d.get("followers")))

    following = max(following, _safe_int(d.get("following")), _safe_int(d.get("friends")))

    # GraphQL-style nested counts (common on Instagram APIs)
    if posts == 0:
        em = d.get("edge_owner_to_timeline_media")
        if isinstance(em, dict):
            posts = _safe_int(em.get("count"))
    if followers == 0:
        ef = d.get("edge_followed_by")
        if isinstance(ef, dict):
            followers = _safe_int(ef.get("count"))
    if following == 0:
        eg = d.get("edge_follow")
        if isinstance(eg, dict):
            following = _safe_int(eg.get("count"))

    return posts, followers, following


def _iter_nested_dicts(obj: Any, depth: int = 0) -> Any:
    if depth > 14:
        return
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_nested_dicts(v, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_nested_dicts(item, depth + 1)


def _best_counts_from_tree(raw: Any) -> tuple[int, int, int]:
    """When the API nests fields unpredictably, pick the dict with the strongest count signal."""
    best = (0, 0, 0)
    best_score = -1
    for d in _iter_nested_dicts(raw):
        p, f, g = _extract_social_counts(d)
        score = p + f + g
        if score > best_score:
            best_score = score
            best = (p, f, g)
    return best


def _rapidapi_parse_json(response: requests.Response) -> Any:
    text = response.text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"RapidAPI: non-JSON body (first 400 chars): {text[:400]!r}")
        return None


def _rapidapi_error_payload(raw: Any) -> Optional[str]:
    if not isinstance(raw, dict):
        return None
    if raw.get("success") is False and raw.get("message"):
        return str(raw["message"])
    if raw.get("error") and isinstance(raw.get("message"), str):
        return str(raw["message"])
    if isinstance(raw.get("msg"), str) and "error" in str(raw).lower():
        return str(raw["msg"])
    return None


def fetch_live_profile(username: str) -> Optional[dict]:
    """
    Fetch profile metrics via RapidAPI (Instagram-oriented endpoint).
    Uses `.env` (RAPIDAPI_KEY) or environment variables; optional RAPIDAPI_HOST, RAPIDAPI_PROFILE_URL.
    """
    if not username:
        return None
    if not RAPIDAPI_KEY:
        print("Live fetch skipped: set RAPIDAPI_KEY in .env or the environment (see .env.example).")
        return None

    form_headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    get_headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
    }

    attempts: list[tuple[str, dict[str, Any]]] = [
        ("POST form", {"method": "POST", "headers": form_headers, "data": {"username_or_url": username}}),
        ("POST + username", {"method": "POST", "headers": form_headers, "data": {"username_or_url": username, "username": username}}),
        ("GET query", {"method": "GET", "headers": get_headers, "params": {"username_or_url": username}}),
        ("GET username", {"method": "GET", "headers": get_headers, "params": {"username": username}}),
    ]

    last_error: Optional[Exception] = None
    for label, kwargs in attempts:
        try:
            method = kwargs.pop("method")
            r = requests.request(method, RAPIDAPI_PROFILE_URL, timeout=45, **kwargs)
            if r.status_code >= 400:
                print(f"RapidAPI {label}: HTTP {r.status_code} for {username!r}")
                last_error = RuntimeError(f"HTTP {r.status_code}")
                continue
            raw = _rapidapi_parse_json(r)
            if raw is None:
                continue

            err = _rapidapi_error_payload(raw)
            if err:
                print(f"RapidAPI {label} error payload: {err}")
                last_error = RuntimeError(err)
                continue

            data = _unwrap_profile_payload(raw)
            posts, followers, following = _extract_social_counts(data) if isinstance(data, dict) else (0, 0, 0)
            alt = _best_counts_from_tree(raw)
            if sum((posts, followers, following)) < sum(alt):
                posts, followers, following = alt

            if posts == 0 and followers == 0 and following == 0:
                print(f"RapidAPI {label}: no counts parsed for {username!r}; sample keys: {_sample_keys(raw)}")
                last_error = RuntimeError("no counts in response")
                continue

            return {
                "statuses_count": posts,
                "followers_count": followers,
                "friends_count": following,
                "favourites_count": 0,
                "listed_count": 0,
                "lang": "en",
            }
        except requests.RequestException as e:
            last_error = e
            print(f"RapidAPI {label} request failed: {e}")
        except Exception as e:
            last_error = e
            print(f"RapidAPI {label} failed for {username}: {e}")

    if last_error:
        print(f"RapidAPI: all attempts failed for {username!r}: {last_error}")
    return None


def _sample_keys(raw: Any, max_len: int = 120) -> str:
    if isinstance(raw, dict):
        return str(list(raw.keys())[:12])
    if isinstance(raw, list) and raw:
        return f"list[len={len(raw)}] first_type={type(raw[0]).__name__}"
    return type(raw).__name__

# Initialize Models on Startup
for key, path in MODEL_PATHS.items():
    ARTIFACTS[key] = load_artifact(path)

def generate_explanation(features: dict, label: str) -> list[str]:
    reasons = []
    statuses = float(features.get('statuses', 0))
    followers = float(features.get('followers', 0))
    friends = float(features.get('friends', 0))
    favourites = float(features.get('favourites', 0))
    listed = float(features.get('listed', 0))
    
    if label == "Fake":
        if followers < 10 and friends > 100:
            reasons.append("Suspicious follower-to-following ratio (very few followers but follows many).")
        if statuses == 0:
            reasons.append("The account has never posted a status/post.")
        if favourites == 0 and listed == 0 and followers < 50:
            reasons.append("The account shows very low overall engagement metrics.")
        if friends > 5000 and followers < 500:
            reasons.append("Extremely high number of friends (following) compared to followers, typical of spam accounts.")
        if not reasons:
            reasons.append("The model detected anomalous feature combinations commonly found in fake or spam accounts.")
    else:
        if followers > 100 and followers >= (friends * 0.5):
            reasons.append("Healthy follower-to-following ratio.")
        if statuses > 10:
            reasons.append("Account has an active posting history.")
        if favourites > 10 or listed > 0:
            reasons.append("Account shows normal engagement with others.")
        if not reasons:
            reasons.append("The profile features match the usual patterns of genuine accounts.")
            
    return reasons

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('dashboard', model='rf'))  # Default to RF Model
    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    # Model Selector logic (Defaults to 'rf')
    selected_model = request.args.get('model', 'rf')
    active_tab = request.args.get('active_tab', 'single')
    if selected_model not in ARTIFACTS:
        selected_model = 'rf'
        
    artifact = ARTIFACTS[selected_model]
    
    prediction_result = None
    confidence = None
    
    # Handle single prediction form submission
    if request.method == 'POST':
        if artifact is None:
            flash(f"Model ({selected_model.upper()}) is not loaded. Please train it first.", "error")
        else:
            try:
                statuses = float(request.form.get('statuses', 0))
                followers = float(request.form.get('followers', 0))
                friends = float(request.form.get('friends', 0))
                favourites = float(request.form.get('favourites', 0))
                listed = float(request.form.get('listed', 0))
                
                lang = request.form.get('lang', 'en')
                lang_code = 0
                if artifact.label_encoder is not None:
                    try:
                        lang_code = int(artifact.label_encoder.transform([lang])[0])
                    except ValueError:
                        flash(f"Unknown language '{lang}', defaulting to code 0", "warning")
                        lang_code = 0
                        
                X = np.array([[statuses, followers, friends, favourites, listed, lang_code]], dtype=float)
                pred, proba = _predict(artifact, X)
                
                label = _label(int(pred[0]))
                
                features_dict = {
                    'statuses': statuses,
                    'followers': followers,
                    'friends': friends,
                    'favourites': favourites,
                    'listed': listed
                }
                
                prediction_result = {
                    "label": label,
                    "is_genuine": label == "Genuine",
                    "explanations": generate_explanation(features_dict, label)
                }
                
                if proba is not None and proba.shape[1] >= 2:
                    p_genuine = float(proba[0][1])
                    p_fake = float(proba[0][0])
                    confidence = {
                        "genuine": round(p_genuine * 100, 2),
                        "fake": round(p_fake * 100, 2)
                    }
                    
            except Exception as e:
                flash(f"Error making prediction: {str(e)}", "error")

    model_status = {k: "Ready" if v is not None else "Not Trained" for k, v in ARTIFACTS.items()}
    
    return render_template('dashboard.html', 
                           artifact=artifact, 
                           result=prediction_result, 
                           confidence=confidence,
                           selected_model=selected_model,
                           model_status=model_status,
                           active_tab=active_tab)

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    selected_model = request.form.get('model_type', 'rf')
    artifact = ARTIFACTS.get(selected_model)
    
    if artifact is None:
        flash(f"Model ({selected_model.upper()}) is not loaded. Please train it first.", "error")
        return redirect(url_for('dashboard', model=selected_model))
        
    if 'csv_file' not in request.files:
        flash("No file part", "error")
        return redirect(url_for('dashboard', model=selected_model))
        
    file = request.files['csv_file']
    if file.filename == '':
        flash("No selected file", "error")
        return redirect(url_for('dashboard', model=selected_model))
        
    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            df = _normalize_columns(df)
            
            if "lang_code" not in df.columns and "lang" in df.columns and artifact.label_encoder is not None:
                known = set(artifact.label_encoder.classes_)
                df["lang"] = df["lang"].astype(str).fillna("")
                df = df[df["lang"].isin(known)].copy()
                if len(df) == 0:
                    flash("No rows left after filtering unknown languages.", "error")
                    return redirect(url_for('dashboard', model=selected_model))
                df["lang_code"] = artifact.label_encoder.transform(df["lang"])

            df = _coerce_numeric_features(df)
            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                flash("Missing required columns: " + ", ".join(missing), "error")
                return redirect(url_for('dashboard', model=selected_model))

            Xdf = df[FEATURES].copy()
            valid = ~Xdf.isna().any(axis=1)
            Xdf = Xdf.loc[valid]
            
            if len(Xdf) == 0:
                flash("No valid rows to predict.", "error")
                return redirect(url_for('dashboard', model=selected_model))

            pred, proba = _predict(artifact, Xdf.to_numpy(dtype=float))
            out = df.loc[Xdf.index].copy()
            out["prediction"] = [int(p) for p in pred]
            out["label"] = [_label(int(p)) for p in pred]

            if proba is not None and proba.shape[1] >= 2:
                out["p_fake"] = proba[:, 0]
                out["p_genuine"] = proba[:, 1]
                
            
            # Convert first 100 rows to dictionary for HTML rendering
            batch_results = out.head(100).to_dict(orient='records')
            
            # Generate the CSV Download Link in memory
            output = io.StringIO()
            out.to_csv(output, index=False)
            csv_data = output.getvalue()
            import base64
            b64 = base64.b64encode(csv_data.encode()).decode()
            csv_download_url = f"data:text/csv;base64,{b64}"
            
            return render_template('dashboard.html', 
                                 artifact=artifact,
                                 selected_model=selected_model,
                                 model_status={k: "Ready" if v is not None else "Not Trained" for k, v in ARTIFACTS.items()},
                                 batch_results=batch_results,
                                 csv_download_url=csv_download_url,
                                 active_tab='batch')
            
            
        except Exception as e:
            flash(f"Error processing CSV: {str(e)}", "error")
            
    return redirect(url_for('dashboard', model=selected_model, active_tab='batch'))

def fetch_recent_image_post(username: str) -> Optional[str]:
    """Fetch the latest image post from a user to test it for AI generation."""
    url = "https://instagram-scraper-stable-api.p.rapidapi.com/get_ig_user_posts.php"
    api_key = os.environ.get("RAPIDAPI_KEY", "d844e8de96msh40ae08e1bca2793p192ff2jsn88b39e227b8e").strip()
    
    payload = f"username_or_url=https://www.instagram.com/{username}/&pagination_token=&amount=3"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'x-rapidapi-host': 'instagram-scraper-stable-api.p.rapidapi.com',
        'x-rapidapi-key': api_key
    }
    try:
        r = requests.post(url, data=payload, headers=headers, timeout=20)
        data = r.json()
        if "data" in data and "items" in data["data"]:
            for item in data["data"]["items"]:
                # Grab a static image (media_type 1 is usually a photo on IG)
                if item.get("media_type") == 1:
                    img_url = item.get("thumbnail_url")
                    if not img_url and "image_versions2" in item:
                        cands = item["image_versions2"].get("candidates", [])
                        if cands:
                            img_url = cands[0].get("url")
                    if img_url:
                        return img_url
    except Exception as e:
        print(f"Error fetching rapidapi posts for AI check: {e}")
    return None

def check_ai_generated_image(image_url: str) -> bool:
    """Passes an image URL to the AI detection API"""
    url = "https://ai-generated-image-detection-api.p.rapidapi.com/v1/image/detect-ai-image"
    api_key = os.environ.get("RAPIDAPI_KEY", "d844e8de96msh40ae08e1bca2793p192ff2jsn88b39e227b8e").strip()
    payload = {"type": "url", "url": image_url}
    headers = {
        'Content-Type': 'application/json',
        'x-rapidapi-host': 'ai-generated-image-detection-api.p.rapidapi.com',
        'x-rapidapi-key': api_key
    }
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=20)
        data = r.json()
        
        # Determine from various possible return structures if it's considered AI
        res_str = str(data).lower()
        if "prediction" in data and "ai" in str(data["prediction"]).lower():
            return True
        if data.get("is_ai_generated") is True or data.get("ai_generated") is True:
            return True
        if data.get("ai_score", 0) > 0.85 or data.get("fake_percentage", 0) > 85:
            return True
        if "ai_detected" in res_str or "artificial" in res_str:
            return True
    except Exception as e:
        print(f"Error checking AI image: {e}")
    return False

@app.route('/predict_realtime', methods=['POST'])
def predict_realtime():
    username = request.form.get('username', '').strip()
    if username.startswith('@'):
        username = username[1:]
        
    selected_model = request.form.get('model', 'rf')
    artifact = ARTIFACTS.get(selected_model)
    
    if not artifact:
        flash(f"Model ({selected_model.upper()}) is not loaded. Please train it first.", "error")
        return redirect(url_for('dashboard', model=selected_model, active_tab='live'))

    if not RAPIDAPI_KEY:
        flash(
            "Live prediction needs a RapidAPI key. Set the RAPIDAPI_KEY environment variable and restart the app.",
            "error",
        )
        return redirect(url_for('dashboard', model=selected_model, active_tab='live'))

    live_features = fetch_live_profile(username)
    if not live_features:
        flash(
            f"Could not load profile data for @{username}. The API may be down, the username may be invalid, or the response format changed. Check the server log for details.",
            "error",
        )
        return redirect(url_for('dashboard', model=selected_model, active_tab='live'))
        
    try:
        lang = live_features['lang']
        lang_code = 0
        if artifact.label_encoder is not None:
            try:
                lang_code = int(artifact.label_encoder.transform([lang])[0])
            except ValueError:
                lang_code = 0
                
        X_live = np.array([[
            live_features['statuses_count'],
            live_features['followers_count'],
            live_features['friends_count'],
            live_features['favourites_count'],
            live_features['listed_count'],
            lang_code
        ]], dtype=float)
        
        pred, proba = _predict(artifact, X_live)
        label = _label(int(pred[0]))
        
        # --- NEW AI IMAGE DETECTION LOGIC ---
        ai_flagged = False
        img_url = fetch_recent_image_post(username)
        if img_url:
            if check_ai_generated_image(img_url):
                ai_flagged = True
                label = "Fake"  # OVERRIDE the tabular model prediction
                
        features_dict = {
            'statuses': live_features['statuses_count'],
            'followers': live_features['followers_count'],
            'friends': live_features['friends_count'],
            'favourites': live_features['favourites_count'],
            'listed': live_features['listed_count']
        }
        
        explanations = generate_explanation(features_dict, label)
        if ai_flagged:
             explanations.insert(0, "CRITICAL: The user's recent profile post photo was analyzed and clearly detected as an AI-GENERATED image!")
        
        prediction_result = {
            "label": label,
            "is_genuine": label == "Genuine",
            "explanations": explanations
        }
        
        confidence = None
        if proba is not None and proba.shape[1] >= 2:
            p_genuine = float(proba[0][1])
            p_fake = float(proba[0][0])
            confidence = {
                "genuine": round(p_genuine * 100, 2),
                "fake": round(p_fake * 100, 2)
            }
            
        model_status = {k: "Ready" if v is not None else "Not Trained" for k, v in ARTIFACTS.items()}
        
        return render_template('dashboard.html',
                               artifact=artifact,
                               result=prediction_result,
                               confidence=confidence,
                               selected_model=selected_model,
                               model_status=model_status,
                               active_tab='live',
                               live_username=username,
                               live_features=live_features)
                               
    except Exception as e:
        flash(f"Error during real-time prediction: {str(e)}", "error")
        return redirect(url_for('dashboard', model=selected_model, active_tab='live'))

@app.route('/train', methods=['POST'])
def train():
    selected_model = request.form.get('model_type', 'rf')
    
    users_path = os.path.join("data", "users.csv")
    fusers_path = os.path.join("data", "fusers.csv")
    
    if not os.path.exists(users_path) or not os.path.exists(fusers_path):
        flash("Training data (users.csv, fusers.csv) not found in data/ folder.", "error")
        return redirect(url_for('dashboard', model=selected_model))

    users = pd.read_csv(users_path)
    fusers = pd.read_csv(fusers_path)

    users["target"] = 1
    fusers["target"] = 0

    df = pd.concat([users, fusers], ignore_index=True)
    le = LabelEncoder()
    df["lang"] = df["lang"].astype(str).fillna("")
    df["lang_code"] = le.fit_transform(df["lang"])

    X = _coerce_numeric_features(df)[FEATURES]
    y = df["target"].astype(int)

    keep = ~X.isna().any(axis=1)
    X = X.loc[keep]
    y = y.loc[keep]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if selected_model == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    elif selected_model == 'svm':
        model = SVC(kernel='rbf', probability=True, random_state=42, class_weight="balanced")
    elif selected_model == 'nn':
        model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=500, random_state=42)
    else:
        flash("Invalid model type.", "error")
        return redirect(url_for('dashboard', model='rf'))

    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        artifact = ModelArtifact(model=model, label_encoder=le, feature_names=tuple(FEATURES))
        joblib.dump(
            {"model": model, "label_encoder": le, "feature_names": list(FEATURES)},
            MODEL_PATHS[selected_model],
        )
        
        # Update global memory
        ARTIFACTS[selected_model] = artifact
        
        flash(f"Successfully trained {selected_model.upper()}! Accuracy: {round(acc*100, 2)}%", "success")
        
    except Exception as e:
        flash(f"Error training model: {str(e)}", "error")

    return redirect(url_for('dashboard', model=selected_model))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
