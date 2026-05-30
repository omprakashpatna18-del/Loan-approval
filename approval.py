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
  cibil:int
  

