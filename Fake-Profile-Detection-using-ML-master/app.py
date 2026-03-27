from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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

MODEL_PATH = "rf_model.pkl"


@dataclass(frozen=True)
class ModelArtifact:
    model: Any
    label_encoder: Optional[LabelEncoder] = None
    feature_names: tuple[str, ...] = tuple(FEATURES)


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


@st.cache_resource(show_spinner=False)
def load_artifact(model_path: str = MODEL_PATH) -> Optional[ModelArtifact]:
    if not os.path.exists(model_path):
        return None

    raw = joblib.load(model_path)

    # Backwards compatibility: allow plain sklearn model pickles
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


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return int(v)
    except Exception:
        return default


def train_and_save_model(model_path: str = MODEL_PATH) -> tuple[ModelArtifact, dict[str, Any]]:
    users_path = os.path.join("data", "users.csv")
    fusers_path = os.path.join("data", "fusers.csv")

    users = pd.read_csv(users_path)
    fusers = pd.read_csv(fusers_path)

    # Label convention: 1 = genuine (users), 0 = fake (fusers)
    users["target"] = 1
    fusers["target"] = 0

    df = pd.concat([users, fusers], ignore_index=True)

    # Build lang_code from the raw 'lang' column (matches typical LabelEncoder behavior)
    le = LabelEncoder()
    df["lang"] = df["lang"].astype(str).fillna("")
    df["lang_code"] = le.fit_transform(df["lang"])

    X = _coerce_numeric_features(df)[FEATURES]
    y = df["target"].astype(int)

    # Basic cleaning: drop rows with missing numeric features
    keep = ~X.isna().any(axis=1)
    X = X.loc[keep]
    y = y.loc[keep]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    artifact = ModelArtifact(model=model, label_encoder=le, feature_names=tuple(FEATURES))
    joblib.dump(
        {"model": model, "label_encoder": le, "feature_names": list(FEATURES)},
        model_path,
    )

    metrics = {"classification_report": report, "confusion_matrix": cm, "rows": int(len(df))}
    return artifact, metrics


st.set_page_config(page_title="Fake Profile Detection", layout="centered")

st.title("Fake Profile Detection")
st.caption("Predict whether a profile looks fake or genuine from basic account statistics.")

with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose a mode", ["Single prediction", "Batch (CSV)", "Train model"], index=0)

    st.divider()
    st.subheader("Model")
    artifact = load_artifact(MODEL_PATH)
    if artifact is None:
        st.warning("Model not found: `rf_model.pkl`")
    else:
        st.success("Loaded: `rf_model.pkl`")


if mode == "Train model":
    st.subheader("Train a Random Forest model")
    st.write("This will train from `data/users.csv` (genuine) and `data/fusers.csv` (fake) and save `rf_model.pkl`.")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        train_clicked = st.button("Train & Save", type="primary", use_container_width=True)
    with col_b:
        st.info("After training, switch back to **Single prediction** or **Batch (CSV)**.")

    if train_clicked:
        with st.spinner("Training model..."):
            artifact, metrics = train_and_save_model(MODEL_PATH)
            load_artifact.clear()  # refresh cache

        st.success("Training complete. Saved `rf_model.pkl`.")
        st.write("Confusion matrix (rows=true, cols=pred):")
        st.json(metrics["confusion_matrix"])
        st.write("Classification report:")
        st.json(metrics["classification_report"])

    st.stop()


if artifact is None:
    st.error("No model available. Train one in the sidebar (Train model), or add `rf_model.pkl` to the project root.")
    st.stop()


def render_language_input(artifact_: ModelArtifact) -> tuple[Optional[str], Optional[int]]:
    if artifact_.label_encoder is None:
        lang_code = st.number_input("Language Code", min_value=0, value=0, help="Numeric encoding used during training.")
        return None, _safe_int(lang_code, 0)

    langs = list(artifact_.label_encoder.classes_)
    default_idx = langs.index("en") if "en" in langs else 0
    lang = st.selectbox("Language", options=langs, index=default_idx, help="Language from the training dataset.")
    lang_code = int(artifact_.label_encoder.transform([lang])[0])
    return lang, lang_code


if mode == "Single prediction":
    st.subheader("Single prediction")
    st.write("Enter profile stats, then predict.")

    c1, c2 = st.columns(2)
    with c1:
        statuses = st.number_input("Statuses Count", min_value=0, value=0, step=1, help="Total posts/statuses.")
        friends = st.number_input("Friends Count", min_value=0, value=0, step=1, help="How many accounts they follow.")
        listed = st.number_input("Listed Count", min_value=0, value=0, step=1, help="Times added to lists.")
    with c2:
        followers = st.number_input("Followers Count", min_value=0, value=0, step=1, help="How many followers.")
        favourites = st.number_input("Favourites Count", min_value=0, value=0, step=1, help="How many likes/favorites.")
        lang, lang_code = render_language_input(artifact)

    X = np.array([[statuses, followers, friends, favourites, listed, lang_code]], dtype=float)

    if st.button("Predict", type="primary"):
        pred, proba = _predict(artifact, X)
        label = _label(int(pred[0]))

        if label == "Genuine":
            st.success("Genuine profile")
        else:
            st.error("Fake profile")

        if proba is not None and proba.shape[1] >= 2:
            p_genuine = float(proba[0][1])
            p_fake = float(proba[0][0])
            st.metric("Confidence (Genuine)", f"{p_genuine:.2%}")
            st.progress(min(max(p_genuine, 0.0), 1.0))
            with st.expander("Details"):
                st.write({"p_fake": round(p_fake, 6), "p_genuine": round(p_genuine, 6), "lang": lang, "lang_code": lang_code})

        if hasattr(artifact.model, "feature_importances_"):
            fi = pd.Series(getattr(artifact.model, "feature_importances_"), index=list(artifact.feature_names))
            st.caption("Top features (global importance)")
            st.bar_chart(fi.sort_values(ascending=False))


if mode == "Batch (CSV)":
    st.subheader("Batch prediction (CSV upload)")
    st.write("Upload a CSV with the feature columns and get predictions + probabilities.")

    st.caption("Expected columns (case-insensitive): " + ", ".join(FEATURES))
    st.caption("Also accepted aliases: " + ", ".join(f"`{v}`→`{k}`" for k, v in CSV_FEATURE_ALIASES.items()))

    up = st.file_uploader("CSV file", type=["csv"])
    if up is None:
        st.stop()

    df = pd.read_csv(up)
    df = _normalize_columns(df)

    # If lang_code isn't present but lang is, build lang_code using the saved encoder.
    if "lang_code" not in df.columns and "lang" in df.columns and artifact.label_encoder is not None:
        known = set(artifact.label_encoder.classes_)
        df["lang"] = df["lang"].astype(str).fillna("")
        unknown_langs = sorted(set(df["lang"]) - known)
        if unknown_langs:
            st.warning(
                "Some `lang` values are not in the model’s encoder; those rows will be dropped.",
            )
            with st.expander("Unknown languages"):
                st.write(unknown_langs[:200])
            df = df[df["lang"].isin(known)].copy()

        if len(df) == 0:
            st.error("No rows left after filtering unknown languages.")
            st.stop()

        df["lang_code"] = artifact.label_encoder.transform(df["lang"])

    df = _coerce_numeric_features(df)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.stop()

    Xdf = df[FEATURES].copy()
    valid = ~Xdf.isna().any(axis=1)
    dropped = int((~valid).sum())
    if dropped:
        st.warning(f"Dropping {dropped} rows with missing/invalid numeric values.")
    Xdf = Xdf.loc[valid]

    if len(Xdf) == 0:
        st.error("No valid rows to predict.")
        st.stop()

    pred, proba = _predict(artifact, Xdf.to_numpy(dtype=float))
    out = df.loc[Xdf.index].copy()
    out["prediction"] = [int(p) for p in pred]
    out["label"] = [_label(int(p)) for p in pred]

    if proba is not None and proba.shape[1] >= 2:
        out["p_fake"] = proba[:, 0]
        out["p_genuine"] = proba[:, 1]

    st.dataframe(out.head(50), use_container_width=True)

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions CSV",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv",
        type="primary",
    )
