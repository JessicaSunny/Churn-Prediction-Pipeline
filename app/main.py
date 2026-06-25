"""
FastAPI app exposing the churn model as a prediction service.

Run locally with: uvicorn app.main:app --reload
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.pkl"

app = FastAPI(title="Churn Prediction API", version="1.0.0")

_artifact = joblib.load(MODEL_PATH)
_pipeline = _artifact["pipeline"]
_features = _artifact["features"]


class CustomerInput(BaseModel):
    gender: str = Field(examples=["Female"])
    SeniorCitizen: int = Field(examples=[0])
    Partner: str = Field(examples=["Yes"])
    Dependents: str = Field(examples=["No"])
    tenure: int = Field(examples=[12])
    PhoneService: str = Field(examples=["Yes"])
    MultipleLines: str = Field(examples=["No"])
    InternetService: str = Field(examples=["Fiber optic"])
    OnlineSecurity: str = Field(examples=["No"])
    OnlineBackup: str = Field(examples=["No"])
    DeviceProtection: str = Field(examples=["No"])
    TechSupport: str = Field(examples=["No"])
    StreamingTV: str = Field(examples=["Yes"])
    StreamingMovies: str = Field(examples=["Yes"])
    Contract: str = Field(examples=["Month-to-month"])
    PaperlessBilling: str = Field(examples=["Yes"])
    PaymentMethod: str = Field(examples=["Electronic check"])
    MonthlyCharges: float = Field(examples=[85.5])
    TotalCharges: float = Field(examples=[1020.0])


class ChurnPrediction(BaseModel):
    churn_probability: float
    churn_risk: str


@app.get("/health")
def health():
    """Used by AWS / load balancers to check the service is alive."""
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnPrediction)
def predict(customer: CustomerInput):
    row = pd.DataFrame([customer.model_dump()])[_features]
    prob = float(_pipeline.predict_proba(row)[0, 1])
    risk = "high" if prob >= 0.5 else "medium" if prob >= 0.25 else "low"
    return ChurnPrediction(churn_probability=round(prob, 4), churn_risk=risk)
