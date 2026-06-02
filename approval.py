import uvicorn
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
  return pd.DataFrame([raw])
@app.post("/predict")
def predict(data:Loan):
  model_df=helper(data)
  predict=model.predict(model_df)[0]
  return predict
    
    

