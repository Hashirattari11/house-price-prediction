import numpy as np
import pickle
import os

MODEL_PATH = "house_rf_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

model = pickle.load(open(MODEL_PATH, "rb"))

def predict_price(bedrooms: float, bathrooms: float, floors: float, sqft_living: float) -> str:
    features = np.array([[bedrooms, bathrooms, floors, sqft_living]])
    prediction = model.predict(features)[0]
    prediction = round(prediction, 2)
    return f"Predicted House Price: ${prediction:,.2f}"
