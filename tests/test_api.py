"""
Automated tests for the churn prediction API.
Run with: pytest
These same tests run automatically in CI on every push (see .github/workflows/ci.yml)
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5,
    "TotalCharges": 191.0,
}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 204
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_probability():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "churn_probability" in body
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_risk"] in {"low", "medium", "high"}


def test_predict_high_risk_profile_flagged_correctly():
    # Short tenure + month-to-month + high charges should score as elevated risk.
    # This guards against silent regressions in the model or preprocessing.
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.json()["churn_probability"] > 0.5


def test_predict_rejects_malformed_input():
    bad_payload = {"gender": "Female"}  # missing required fields
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
