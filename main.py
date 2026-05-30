from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
import joblib
import numpy as np
import logging
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = FastAPI()

load_model = None
load_scaler = None
app.mount("/static", StaticFiles(directory="static"), name="static")

class wineModel(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

    @field_validator("*")
    @classmethod
    def must_not_be_negative(cls, v, info):
        if v < 0:
            raise ValueError(f"{info.field_name} must not be a negative number")
        return v


@app.on_event("startup")
async def load_artifacts():
    global load_model, load_scaler
    model_path = os.getenv("MODEL_PATH")
    scaler_path = os.getenv("SCALER_PATH")
    if not model_path or not scaler_path:
        logging.error("Model path or scaler path not found in environment variables.")
        return
    try:
        load_model = joblib.load(model_path)
        load_scaler = joblib.load(scaler_path)
        logging.info(" Model and scaler loaded successfully.")
    except Exception as e:
        logging.error(f"Error occurred while loading artifacts: {e}")


@app.get('/')
def read_root():
    return FileResponse('static/index.html')


@app.get("/predict")
async def predict_page():
    return FileResponse('static/predict.html')

@app.post('/predict')
def pred_quality(data: wineModel):
    if load_scaler is None or load_model is None:
        logging.error("Model and scaler artifacts not loaded.")
        raise HTTPException(status_code=503, detail="Model and scaler artifacts not loaded.")
    features = np.array([[data.fixed_acidity, data.volatile_acidity, data.citric_acid, data.residual_sugar, data.chlorides, data.free_sulfur_dioxide, data.total_sulfur_dioxide, data.density, data.pH, data.sulphates, data.alcohol]])
    scaled_features = load_scaler.transform(features)
    try:
        pred = load_model.predict(scaled_features)
        logging.info(f"Prediction made successfully: {pred[0]}")
    except Exception as e:
        logging.error(f"Error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error occurred during prediction.")
    return {"predicted_quality": str(pred[0])}
