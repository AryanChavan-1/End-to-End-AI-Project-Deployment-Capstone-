"""
Customer Churn Prediction — Model Training & Serialization

Trains a Scikit-Learn pipeline (preprocessing + RandomForestClassifier)
on the Telco Customer Churn dataset and serializes the full pipeline
as a single .pkl file using joblib.

If `data/Telco-Customer-Churn.csv` exists, uses real data.
Otherwise, generates synthetic data with identical schema.

Usage:
    python model/train.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco-Customer-Churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
PIPELINE_PATH = os.path.join(MODEL_DIR, "churn_pipeline.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")

# ---------------------------------------------------------------------------
# Feature Definitions
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Churn"


# ---------------------------------------------------------------------------
# Synthetic Data Generator
# ---------------------------------------------------------------------------
def generate_synthetic_data(n_samples: int = 7000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic data matching the Telco Customer Churn schema."""
    rng = np.random.RandomState(seed)

    data = {
        "customerID": [f"CUST-{i:05d}" for i in range(n_samples)],
        "gender": rng.choice(["Male", "Female"], n_samples),
        "SeniorCitizen": rng.choice([0, 1], n_samples, p=[0.84, 0.16]),
        "Partner": rng.choice(["Yes", "No"], n_samples),
        "Dependents": rng.choice(["Yes", "No"], n_samples, p=[0.30, 0.70]),
        "tenure": rng.randint(0, 73, n_samples),
        "PhoneService": rng.choice(["Yes", "No"], n_samples, p=[0.90, 0.10]),
        "InternetService": rng.choice(
            ["DSL", "Fiber optic", "No"], n_samples, p=[0.34, 0.44, 0.22]
        ),
        "Contract": rng.choice(
            ["Month-to-month", "One year", "Two year"],
            n_samples,
            p=[0.55, 0.21, 0.24],
        ),
        "PaperlessBilling": rng.choice(["Yes", "No"], n_samples, p=[0.60, 0.40]),
        "PaymentMethod": rng.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            n_samples,
        ),
        "MonthlyCharges": np.round(rng.uniform(18.25, 118.75, n_samples), 2),
    }

    df = pd.DataFrame(data)

    # Phone-dependent feature
    df["MultipleLines"] = df.apply(
        lambda r: "No phone service"
        if r["PhoneService"] == "No"
        else rng.choice(["Yes", "No"]),
        axis=1,
    )

    # Internet-dependent features
    internet_dependent = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for feat in internet_dependent:
        df[feat] = df.apply(
            lambda r: "No internet service"
            if r["InternetService"] == "No"
            else rng.choice(["Yes", "No"]),
            axis=1,
        )

    # TotalCharges = tenure * MonthlyCharges (with small noise)
    df["TotalCharges"] = np.round(
        df["tenure"] * df["MonthlyCharges"] * rng.uniform(0.85, 1.15, n_samples), 2
    )
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0

    # Generate realistic churn labels — higher churn for:
    # month-to-month contracts, fiber optic, electronic check, short tenure
    churn_prob = np.zeros(n_samples)
    churn_prob += (df["Contract"] == "Month-to-month").astype(float) * 0.25
    churn_prob += (df["InternetService"] == "Fiber optic").astype(float) * 0.15
    churn_prob += (df["PaymentMethod"] == "Electronic check").astype(float) * 0.10
    churn_prob += (df["tenure"] < 12).astype(float) * 0.15
    churn_prob += (df["MonthlyCharges"] > 70).astype(float) * 0.10
    churn_prob += (df["TechSupport"] == "No").astype(float) * 0.05
    churn_prob += (df["OnlineSecurity"] == "No").astype(float) * 0.05
    churn_prob = np.clip(churn_prob + rng.normal(0, 0.08, n_samples), 0.05, 0.95)

    df["Churn"] = (rng.random(n_samples) < churn_prob).astype(int)
    df["Churn"] = df["Churn"].map({1: "Yes", 0: "No"})

    return df


# ---------------------------------------------------------------------------
# Data Loading & Cleaning
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load real CSV or generate synthetic data."""
    if os.path.exists(DATA_PATH):
        print(f"[INFO] Loading real dataset from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print("[INFO] Real dataset not found. Generating synthetic data...")
        df = generate_synthetic_data()
        # Save synthetic data for reference
        os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"[INFO] Synthetic data saved to {DATA_PATH}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the data for training."""
    df = df.copy()

    # Drop customer ID
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Fix TotalCharges — whitespace strings to NaN, then fill with 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(0.0, inplace=True)

    # Ensure SeniorCitizen is string for consistent encoding
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # Encode target
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

    return df


# ---------------------------------------------------------------------------
# Pipeline Building
# ---------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    """Build sklearn pipeline with preprocessing + classifier."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return pipeline


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------
def train():
    """Full training pipeline: load → clean → train → evaluate → save."""
    print("=" * 60)
    print("  Customer Churn Prediction — Model Training")
    print("=" * 60)

    # 1. Load data
    df = load_data()
    print(f"[INFO] Dataset shape: {df.shape}")

    # 2. Clean data
    df = clean_data(df)
    print(f"[INFO] Cleaned dataset shape: {df.shape}")
    print(f"[INFO] Churn distribution:\n{df[TARGET].value_counts(normalize=True)}")

    # 3. Split
    X = df[ALL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # 4. Build & train pipeline
    pipeline = build_pipeline()
    print("[INFO] Training model...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Accuracy: {accuracy:.4f}")
    print("\n[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # 6. Serialize
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)
    joblib.dump(ALL_FEATURES, FEATURES_PATH)
    print(f"[INFO] Pipeline saved to {PIPELINE_PATH}")
    print(f"[INFO] Feature names saved to {FEATURES_PATH}")
    print(f"[INFO] Model file size: {os.path.getsize(PIPELINE_PATH) / 1024:.1f} KB")
    print("=" * 60)
    print("  Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    train()
