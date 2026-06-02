import uvicorn
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

model=joblib.load("best_model.joblib")
app = FastAPI(title="Loan Predictor")

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_methods=["*"],
allow_headers=["*"],
                )

class Loan(BaseModel):
  no_of_dependents:int
  loan_amount:float
  cibil_score:int
  education:str
  self_employed:str
  loan_term:int
  residential_assets_value:float
  commercial_assets_value:float
  luxury_assets_value:float
  bank_asset_value:float
  
def helper(data:Loan):
  raw = {
        "no_of_dependents": data.no_of_dependents,
        "loan_amount": data.loan_amount,
        "cibil_score": data.cibil_score,
        "education": data.education.title().strip(),
        "self_employed": data.self_employed.title().strip(),
        "loan_term": data.loan_term,
        "residential_assets_value": data.residential_assets_value,
        "commercial_assets_value": data.commercial_assets_value,
        "luxury_assets_value": data.luxury_assets_value,
        "bank_asset_value": data.bank_asset_value
    }
  return raw
@app.post("/predict")
def predict(data:Loan):
  model_df=helper(data)
  predict=model.predict(model_df)[0]
  return predict
    
    

