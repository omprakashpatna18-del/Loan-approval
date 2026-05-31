import uvicorn
import joblib

import google.genai as genai

from fastapi import FastAPI
from fastapi import APIRouter

from pydantic import BaseModel
model=joblib.load("best_model.joblib")
class Loan(BaseModel):
  no_of_dependents:int
  loan_amount:float
  cibil_score:int
  education:str
  self-employed:str
  loan_term:int
  residential_assets_value:float
  commercial_assets_value:float
  luxury_assets_value:float
  bank_asset_value:float
  
def helper(data:Loan):
  raw={
    "no_of_dependents":data.get("no_of_dependents"),
    "loan_amount":data.get("loan_amount"),
    "cibil_score":data.get("cibil_score"),
  "education":str(data.get("education")).title().strip(),
  "self_employed":str(data.get("self_employed").title().strip(),
  "loan_term":data.get("loan_term"),
  "residential_assets_value":data.get("residential_assets_value"),
  "commercial_assets_value":data.get("commercial_assets_value"),
  "luxury_assets_value":data.get("luxury_assets_value"),
  "bank_asset_value":data.get("bank_asset_value")
                      }
  return model_df
  def predict(model_df):
    predict=model.predict(data)[0]
    return predict
    
    

