import pytest  
from fastapi.testclient import TestClient 
from main import app 

@pytest.fixture
def client():
    ''' Create a TestClient Insatance for the Appp'''
    with TestClient(app) as client:
        yield client 
def test_home(client):
    response = client.get("/")
    assert response.status_code == 200 
    assert "text/html" in response.headers["content-type"]
    assert "Wine Quality Predictor" in response.text

def test_predict_page(client):
    response = client.get("/predict")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Predict Wine Quality" in response.text

def valid_payload():
    ''' A fully valid wineModel request body (all 11 fields positive) '''
    return {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.70,
        "citric_acid": 0.00,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }

WINE_FIELDS = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol",
]

@pytest.mark.parametrize("field", WINE_FIELDS)
@pytest.mark.parametrize("bad_value", [-2.0, -0.001, -9999.0])
def test_predict_rejects_negative_input(client, field, bad_value):
    payload = valid_payload()
    payload[field] = bad_value
    response = client.post("/predict", json=payload)
    assert response.status_code == 422