from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import Optional

app = FastAPI(
    title="Bank Term Deposit Subscription Predictor",
    description="API for predicting client subscription to bank term deposits",
    version="1.0"
)

# Load model with version compatibility check
try:
    model = joblib.load("DTCv2.joblib")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    raise RuntimeError("Model loading failed - check sklearn versions")

class ClientFeatures(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str 
    month: str
    day_of_week: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float

@app.post("/predict")
async def predict_subscription(client: ClientFeatures):
    """
    Predicts whether a client will subscribe to a term deposit
    Returns:
    - prediction: 1 (yes) or 0 (no)
    - probability: Confidence score (0-1)
    - message: Plain English prediction
    """
    try:
        # Convert input to dataframe
        input_data = client.dict()
        input_df = pd.DataFrame([input_data])
        
        # Get prediction
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]  # Probability of "yes"
        
        # Format response
        result = {
            "prediction": int(prediction),
            "probability": float(proba),
            "message": "Will subscribe" if prediction == 1 else "Will not subscribe",
            "confidence": f"{proba:.0%} confidence"
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/")
async def health_check():
    return {
        "status": "operational",
        "model": "Bank Subscription Predictor",
        "version": "1.0"
    }