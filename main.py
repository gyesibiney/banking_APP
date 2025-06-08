from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import Optional

# Define Pydantic models directly in main.py
class PredictionInput(BaseModel):
    age: int
    call_duration: int
    campaign_contacts: int
    days_since_contact: int
    previous_contacts: int
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float
    job: str
    marital: str
    education: str
    loan: str
    contact_type: str

class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    confidence: str

app = FastAPI()

# Load model
try:
    model = joblib.load("DTCv2.joblib")
except Exception as e:
    print(f"Failed to load model: {e}")

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        input_df = pd.DataFrame([input_data.dict()])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        confidence = "high" if proba > 0.7 else "medium" if proba > 0.5 else "low"
        
        return {
            "prediction": int(prediction),
            "probability": float(proba),
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Banking Campaign Prediction API"}