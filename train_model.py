"""
Trains a churn prediction model and saves it to model/model.pkl.

Run with: python train_model.py
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/telco_churn.csv"
MODEL_PATH = "model/model.pkl"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # TotalCharges has occasional blank strings in the real dataset
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def main():
    df = load_data(DATA_PATH)
    X = df[ALL_FEATURES]
    y = (df["Churn"] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, preds))
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.3f}")

    joblib.dump({"pipeline": pipeline, "features": ALL_FEATURES}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
