from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the model
model = joblib.load("best_model.pkl")

# Define the expected input schema
class ClientData(BaseModel):
    Age: float
    Call_Duration_Seconds: float
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

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Bank Subscription Prediction API"}

@app.post("/predict/")
def predict(data: ClientData):
    # Convert to DataFrame
    input_df = pd.DataFrame([data.dict()])
    # Make prediction
    prediction = model.predict(input_df)
    return {"subscribe": bool(prediction[0])}
