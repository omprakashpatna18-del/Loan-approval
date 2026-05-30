import uvicorn
import joblib

import google.genai as genai

from fastapi import FastAPI
from fastapi import APIRouter

from pydantic import BaseModel
model=joblib.load("best_model.joblib")
