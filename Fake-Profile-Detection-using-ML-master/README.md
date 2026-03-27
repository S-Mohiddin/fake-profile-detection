# Fake profile detection (ML)

Web app and notebooks for experimenting with **fake vs genuine** social-profile classification using numeric features (statuses, followers, friends, favourites, listed count, language). Models include **Random Forest**, **SVM**, and a small **neural network**.

## Features

- **Flask UI** (`app_flask.py`): single-profile prediction, batch CSV, optional live fetch (RapidAPI), and training from `data/users.csv` + `data/fusers.csv`.
- **Notebooks** (`.ipynb`): exploratory training and exports to `html/` for reports.
- **Streamlit** (`app.py`): alternate simple UI if you prefer Streamlit.

Results are **probabilistic** and depend on the training data and features you use—they are not a guarantee about any real person or account.

## Screenshots (optional)

To make the GitHub page look polished, add:

- `docs/screenshot-home.png` — landing page  
- `docs/screenshot-dashboard.png` — dashboard  

Then add standard Markdown image lines in this README, for example:

`![Home](docs/screenshot-home.png)`

## Quick start (Flask)

**Requirements:** Python 3.10+ recommended.

```bash
cd Fake-Profile-Detection-using-ML-master
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Run the web app:**

```bash
python app_flask.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Use **Login** (demo) to reach the dashboard, or go directly to `/dashboard` if your routes allow it.

**Optional:** set a secret key for flash messages and sessions:

```bash
set FLASK_SECRET_KEY=your-long-random-string
python app_flask.py
```

(On Linux/macOS use `export FLASK_SECRET_KEY=...`.)

### Live prediction (RapidAPI)

The **Live** tab calls Instagram-oriented endpoints via [RapidAPI](https://rapidapi.com/). Subscribe to the API (e.g. **Instagram Scraper Stable API**), then add your key **without committing it**:

1. Copy `.env.example` to `.env` in the project folder.
2. Set `RAPIDAPI_KEY=` to your RapidAPI key.
3. Restart `python app_flask.py`.

Alternatively use environment variables only:

```bash
set RAPIDAPI_KEY=your_rapidapi_key_here
```

Optional overrides if the provider changes URLs:

- `RAPIDAPI_HOST` — default `instagram-scraper-stable-api.p.rapidapi.com`
- `RAPIDAPI_PROFILE_URL` — default `https://instagram-scraper-stable-api.p.rapidapi.com/ig_get_fb_profile_v3.php`

Restart the Flask app after changing these variables.

## Quick start (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Quick start (notebooks)

```bash
pip install ipython jupyter matplotlib
jupyter notebook
```

Open any `.ipynb` file and run cells. Exported HTML reports live under `html/`.

## Data and models

- Training CSVs are expected under `data/` (e.g. `users.csv`, `fusers.csv`) as used by the **Train** tab in the Flask dashboard.
- Trained weights are saved next to the app as `rf_model.pkl`, `svm_model.pkl`, `nn_model.pkl` after training.

## Original project reference

Freelancer.com project URL: https://www.freelancer.in/projects/Python/Write-some-Software-9045568/

---

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/default-orange.png)](https://www.buymeacoffee.com/cognitivecamp)
