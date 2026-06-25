"""
Generates a SYNTHETIC stand-in for the Telco Customer Churn dataset, matching
its exact column schema. Used only so the pipeline can be built and tested
end-to-end before you plug in the real data.

Replace data/telco_churn.csv with the real dataset from:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
(same column names, so no other code needs to change)
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

df = pd.DataFrame({
    "customerID": [f"CUST-{i:05d}" for i in range(n)],
    "gender": np.random.choice(["Male", "Female"], n),
    "SeniorCitizen": np.random.choice([0, 1], n, p=[0.85, 0.15]),
    "Partner": np.random.choice(["Yes", "No"], n),
    "Dependents": np.random.choice(["Yes", "No"], n, p=[0.3, 0.7]),
    "tenure": np.random.randint(0, 73, n),
    "PhoneService": np.random.choice(["Yes", "No"], n, p=[0.9, 0.1]),
    "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n),
    "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
    "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n),
    "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n),
    "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n),
    "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n),
    "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n),
    "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n),
    "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2]),
    "PaperlessBilling": np.random.choice(["Yes", "No"], n),
    "PaymentMethod": np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
    ),
    "MonthlyCharges": np.round(np.random.uniform(18, 120, n), 2),
})
df["TotalCharges"] = np.round(df["tenure"] * df["MonthlyCharges"] * np.random.uniform(0.9, 1.0, n), 2)

# Make churn probability depend on real signal (short tenure, month-to-month,
# high charges) so the model actually has something to learn -- not pure noise.
churn_logit = (
    -1.5
    + 1.8 * (df["Contract"] == "Month-to-month")
    - 0.04 * df["tenure"]
    + 0.01 * df["MonthlyCharges"]
    + 0.5 * (df["InternetService"] == "Fiber optic")
)
churn_prob = 1 / (1 + np.exp(-churn_logit))
df["Churn"] = np.where(np.random.uniform(0, 1, n) < churn_prob, "Yes", "No")

df.to_csv("data/telco_churn.csv", index=False)
print(f"Wrote {len(df)} rows to data/telco_churn.csv")
print(df["Churn"].value_counts(normalize=True))
