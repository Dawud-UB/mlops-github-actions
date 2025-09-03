import joblib
import os

def test_model_exists():
    assert os.path.exists("model.pkl"), "Model file not found!"

def test_model_prediction():
    model = joblib.load("model.pkl")
    pred = model.predict([[14.0, 14.0, 90.0, 600.0, 0.1, 0.2, 0.3, 0.1, 0.2,
                           0.05, 0.3, 1.0, 2.0, 25.0, 0.005, 0.01, 0.02, 0.005,
                           0.02, 0.002, 16.0, 20.0, 100.0, 800.0, 0.15, 0.5,
                           0.6, 0.2, 0.3, 0.07]])
    assert pred is not None
