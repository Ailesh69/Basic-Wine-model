from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np 
import pandas as pd 

load_model = joblib.load('model.pkl')
load_scaler = joblib.load('scaler.pkl')

app = FastAPI()

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

@app.get('/')
def read_root():
    return {"message": "Welcome to the Wine Quality Prediction API"}

@app.post('/predict')
def pred_quality(data: wineModel):
    features = np.array([[data.fixed_acidity, data.volatile_acidity, data.citric_acid, data.residual_sugar, data.chlorides, data.free_sulfur_dioxide, data.total_sulfur_dioxide, data.density, data.pH, data.sulphates, data.alcohol]])
    scaled_features = load_scaler.transform(features)
    pred = load_model.predict(scaled_features)
    return {"predicted_quality": str(pred[0])}