# app.py
## Flask | Django | FastAPI

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

### API - Application Programming Interface : A way to communicate between two applications

# === Load Model ===
MODEL_PATH = "/Users/rohanpatil/Desktop/ai_ml_projects/project_1/models/health_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# === FastAPI App ===
app = FastAPI(title="Hearing Test Predictor")

# === Input Schema ===
class PredictionInput(BaseModel):
    age: float
    physical_score: float
    

# === Routes ===
@app.get("/")
def read_root():
    return {"message": "Hearing Test Predictor is running fine !"}

@app.post("/predict/")
def predict(input_data: PredictionInput):
    # Convert input to numpy array
    features = np.array([[input_data.age, input_data.physical_score]])
    # Predict
    prediction = model.predict(features)
    return {"prediction": float(prediction[0])}
