# 🍷 Wine Quality Classification

A machine learning project that predicts the quality of red wine (`good` / `average` / `bad`)
from its physicochemical properties. It includes a trained scikit-learn model, a FastAPI
backend, and a simple HTML/CSS frontend.

## Overview

- **Dataset:** `WineQT.csv` (Wine Quality dataset, 11 physicochemical features per sample).
- **Target:** the original 0–10 `quality` score is bucketed into three classes:
  - `bad` → quality 3–4
  - `average` → quality 5–6
  - `good` → quality 7–8
- **Model:** a `RandomForestClassifier` tuned with `GridSearchCV` (selected as the best
  performer among Logistic Regression, SVM, Decision Tree, KNN, and Naive Bayes).
- **Preprocessing:** features are scaled with `MinMaxScaler`.

## Project structure

```
Wine_classification/
├── WineQT.csv          # Dataset
├── model.py            # Trains models, tunes the best one, saves artifacts
├── model.pkl           # Saved trained model
├── scaler.pkl          # Saved fitted scaler
├── main.py             # FastAPI backend (serves frontend + /predict API)
├── Static/             # Frontend
│   ├── Index.html      # Landing page
│   ├── predict.html    # Prediction form
│   └── style.css       # Styles
├── .env                # MODEL_PATH / SCALER_PATH (not committed)
└── .gitignore
```

## Setup

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install fastapi uvicorn scikit-learn pandas numpy joblib python-dotenv seaborn matplotlib
```
or you can run 
pip install -r requirements.txt

Create a `.env` file in the project root:

```
MODEL_PATH="model.pkl"
SCALER_PATH="scaler.pkl"
```

## Training (optional)

The repo already ships with trained `model.pkl` and `scaler.pkl`. To retrain:

```bash
python model.py
```

This trains and compares several models, runs a grid search on the Random Forest,
prints a classification report, and re-saves `model.pkl` and `scaler.pkl`.

## Running the app

```bash
uvicorn main:app --reload
```

Then open <http://127.0.0.1:8000/> in your browser:

- `/` — landing page
- `/predict` — prediction form (fill in the 11 features and submit)

## API

### `POST /predict`

**Request body (JSON):**

```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.70,
  "citric_acid": 0.00,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11,
  "total_sulfur_dioxide": 34,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

**Response:**

```json
{ "predicted_quality": "average" }
```

Interactive API docs are available at <http://127.0.0.1:8000/docs>.

## Notes

- The `Id` column in the dataset is dropped during training — it is a row identifier,
  not a predictive feature.
- The `bad` class is underrepresented in the data, so predictions for low-quality wines
  are less reliable than for `average`/`good`.
