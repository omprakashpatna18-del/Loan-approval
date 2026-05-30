import uvicorn
import joblib

import google.genai as genai

from fastapi import FastAPI
from fastapi import APIRouter

from pydantic import BaseModel
model=joblib.load("best_model.joblib")
class Loan(BaseModel):
  loan_id:str
  no_of_dependents:int
  loan_amount:float
  cibil_score:int
  loan_amount:float
  education:str
  self-employed:str
  loan_term:int
  residential_assets_value:float
  commercial_assets_value:float
  luxury_assets_value:float
  bank_asset_value:float
  
  

