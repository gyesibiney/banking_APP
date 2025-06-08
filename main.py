from fastapi import FastAPI, Query
from enum import Enum
from typing import Optional
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load("DTCv2.joblib")

# Define Enums for categorical columns
class JobEnum(str, Enum):
    housemaid = "housemaid"
    services = "services"
    admin = "admin."
    blue_collar = "blue-collar"
    technician = "technician"
    retired = "retired"
    management = "management"
    unemployed = "unemployed"
    self_employed = "self-employed"
    entrepreneur = "entrepreneur"
    student = "student"
    

class MaritalEnum(str, Enum):
    married = "married"
    single = "single"
    divorced = "divorced"
    

class EducationEnum(str, Enum):
    basic_4y = "basic.4y"
    basic_6y = "basic.6y"
    basic_9y = "basic.9y"
    high_school = "high.school"
    university_degree = "university.degree"
    professional_course = "professional.course"
    illiterate = "illiterate"
    

class YesNoUnknownEnum(str, Enum):
    yes = "yes"
    no = "no"
    

class ContactEnum(str, Enum):
    telephone = "telephone"
    cellular = "cellular"

class MonthEnum(str, Enum):
    jan = "jan"
    feb = "feb"
    mar = "mar"
    apr = "apr"
    may = "may"
    jun = "jun"
    jul = "jul"
    aug = "aug"
    sep = "sep"
    oct = "oct"
    nov = "nov"
    dec = "dec"

class DayOfWeekEnum(str, Enum):
    mon = "mon"
    tue = "tue"
    wed = "wed"
    thu = "thu"
    fri = "fri"

class PoutcomeEnum(str, Enum):
    nonexistent = "nonexistent"
    failure = "failure"
    success = "success"

class OutputEnum(str, Enum):
    yes = "yes"
    no = "no"

@app.get("/predict")
def predict(
    age: Optional[int] = Query(...),
    duration: Optional[int] = Query(...),
    campaign: Optional[int] = Query(...),
    pdays: Optional[int] = Query(...),
    previous: Optional[int] = Query(...),
    emp_var_rate: Optional[float] = Query(...),
    cons_price_idx: Optional[float] = Query(...),
    cons_conf_idx: Optional[float] = Query(...),
    euribor3m: Optional[float] = Query(...),
    nr_employed: Optional[float] = Query(...),
    job: Optional[JobEnum] = Query(...),
    marital: Optional[MaritalEnum] = Query(...),
    education: Optional[EducationEnum] = Query(...),
    default: Optional[YesNoUnknownEnum] = Query(...),
    housing: Optional[YesNoUnknownEnum] = Query(...),
    loan: Optional[YesNoUnknownEnum] = Query(...),
    contact: Optional[ContactEnum] = Query(...),
    month: Optional[MonthEnum] = Query(...),
    day_of_week: Optional[DayOfWeekEnum] = Query(...),
    poutcome: Optional[PoutcomeEnum] = Query(...)
):
    input_data = pd.DataFrame([{
        "Age": age,
        "Call_Duration_Seconds": duration,
        "Contacts_During_Campaign": campaign,
        "Days_Since_Last_Contact": pdays,
        "Previous_Contacts": previous,
        "Employment_Variation_Rate": emp_var_rate,
        "Consumer_Price_Index": cons_price_idx,
        "Consumer_Confidence_Index": cons_conf_idx,
        "Euribor_3M_Rate": euribor3m,
        "Number_Employed": nr_employed,
        "Job": job,
        "Marital_Status": marital,
        "Education_Level": education,
        "Has_Default": default,
        "Has_Housing_Loan": housing,
        "Has_Personal_Loan": loan,
        "Contact_Type": contact,
        "Contact_Month": month,
        "Contact_Day": day_of_week,
        "Previous_Outcome": poutcome
    }])

    prediction = model.predict(input_data)[0]
    result = "Client will subscribe" if prediction == 1 else "Client will not subscribe"
    return {"prediction": result}

