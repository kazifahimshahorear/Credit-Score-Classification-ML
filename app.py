from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load saved model & feature list
model = joblib.load('best_credit_score_model.pkl')
selected_features = joblib.load('selected_features.pkl')

app = FastAPI(title="Credit Score Classification API")

class CreditData(BaseModel):
    Annual_Income: float
    Monthly_Inhand_Salary: float
    Num_Bank_Accounts: float
    Type_of_Loan: float
    Delay_from_due_date: float
    Num_of_Delayed_Payment: float
    Changed_Credit_Limit: float
    Num_Credit_Inquiries: float
    Credit_Mix: float
    Outstanding_Debt: float
    Credit_Utilization_Ratio: float
    Credit_History_Age: float
    Payment_of_Min_Amount: float
    Amount_invested_monthly: float
    Payment_Behaviour: float

@app.get("/")
def home():
    return {"message": "Credit Score Classification API is running!"}

@app.post("/predict")
def predict_credit_score(data: CreditData):
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])
    
    # Ensure correct column order
    df_filtered = df[selected_features]
    
    prediction = model.predict(df_filtered)[0]
    return {"Predicted_Credit_Score": prediction}