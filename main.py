from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel

app = FastAPI()

# Load your model
model = joblib.load('DTCv2.joblib')

# Define input data structure
class CustomerInput(BaseModel):
    Age: int
    Call_Duration_Seconds: int
    Contacts_During_Campaign: int
    Days_Since_Last_Contact: int
    Previous_Contacts: int
    Employment_Variation_Rate: float
    Consumer_Price_Index: float
    Consumer_Confidence_Index: float
    Euribor_3M_Rate: float
    Number_Employed: float
    Job: str
    Marital_Status: str
    Education_Level: str
    Has_Default: str
    Has_Housing_Loan: str
    Has_Personal_Loan: str
    Contact_Type: str
    Contact_Month: str
    Contact_Day: str
    Previous_Outcome: str

@app.get("/")
def home():
    return {"message": "Banking Churn Prediction API"}

@app.post("/predict")
def predict(input_data: CustomerInput):
    # Convert input to DataFrame
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "prediction": "Will Subscribe" if prediction == 1 else "Will not Subscribe",
        "probability": float(probability)
    }