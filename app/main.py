"""
Customer Churn Prediction API

FastAPI application that serves a trained Scikit-Learn pipeline for
customer churn prediction. Includes Pydantic validation, health checks,
CORS support, and an embedded HTML frontend.

Usage:
    uvicorn app.main:app --reload
"""

import os
import joblib
import pandas as pd
import numpy as np
from enum import Enum
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Feature definitions (must match training)
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

# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(BASE_DIR, "model", "churn_pipeline.pkl")
)


# ---------------------------------------------------------------------------
# Pydantic Enums — strict categorical validation
# ---------------------------------------------------------------------------
class Gender(str, Enum):
    male = "Male"
    female = "Female"


class YesNo(str, Enum):
    yes = "Yes"
    no = "No"


class SeniorCitizenType(str, Enum):
    zero = "0"
    one = "1"


class InternetServiceType(str, Enum):
    dsl = "DSL"
    fiber_optic = "Fiber optic"
    no = "No"


class ContractType(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class PaymentMethodType(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"


class MultipleLinesType(str, Enum):
    yes = "Yes"
    no = "No"
    no_phone = "No phone service"


class InternetDependentService(str, Enum):
    yes = "Yes"
    no = "No"
    no_internet = "No internet service"


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------
class CustomerData(BaseModel):
    """Input schema for churn prediction. All fields are validated."""

    # Demographics
    gender: Gender
    SeniorCitizen: SeniorCitizenType = Field(
        description="0 = Not senior, 1 = Senior citizen"
    )
    Partner: YesNo
    Dependents: YesNo

    # Account Info
    tenure: int = Field(ge=0, le=72, description="Months with company (0-72)")
    Contract: ContractType
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethodType
    MonthlyCharges: float = Field(
        ge=0, le=200, description="Monthly charge amount in USD"
    )
    TotalCharges: float = Field(ge=0, description="Total charges to date in USD")

    # Services
    PhoneService: YesNo
    MultipleLines: MultipleLinesType
    InternetService: InternetServiceType
    OnlineSecurity: InternetDependentService
    OnlineBackup: InternetDependentService
    DeviceProtection: InternetDependentService
    TechSupport: InternetDependentService
    StreamingTV: InternetDependentService
    StreamingMovies: InternetDependentService

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Male",
                    "SeniorCitizen": "0",
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 12,
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 70.35,
                    "TotalCharges": 844.20,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for churn prediction."""

    prediction: str = Field(description="Churn or No Churn")
    churn_probability: float = Field(
        ge=0, le=1, description="Probability of churn (0-1)"
    )
    risk_level: str = Field(description="Low / Medium / High")
    confidence: float = Field(
        ge=0, le=1, description="Model confidence in the prediction"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str


# ---------------------------------------------------------------------------
# App Lifespan — load model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model pipeline at startup, release at shutdown."""
    try:
        app.state.model = joblib.load(MODEL_PATH)
        app.state.model_loaded = True
        print(f"[INFO] Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        app.state.model = None
        app.state.model_loaded = False
        print(f"[WARNING] Model not found at {MODEL_PATH}. Run model/train.py first.")
    yield
    app.state.model = None


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "REST API for predicting customer churn using a trained "
        "Scikit-Learn Random Forest pipeline. Part of the End-to-End "
        "AI Project Deployment Capstone."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request):
    """Health check endpoint. Returns 200 if the service is alive."""
    return HealthResponse(
        status="healthy",
        model_loaded=request.app.state.model_loaded,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: Request, data: CustomerData):
    """
    Predict customer churn.

    Accepts customer data as JSON, runs it through the trained pipeline,
    and returns the churn prediction with probability and risk level.
    """
    if not request.app.state.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first (python model/train.py).",
        )

    model = request.app.state.model

    # Build DataFrame with correct feature order
    input_dict = {
        "tenure": data.tenure,
        "MonthlyCharges": data.MonthlyCharges,
        "TotalCharges": data.TotalCharges,
        "gender": data.gender.value,
        "SeniorCitizen": data.SeniorCitizen.value,
        "Partner": data.Partner.value,
        "Dependents": data.Dependents.value,
        "PhoneService": data.PhoneService.value,
        "MultipleLines": data.MultipleLines.value,
        "InternetService": data.InternetService.value,
        "OnlineSecurity": data.OnlineSecurity.value,
        "OnlineBackup": data.OnlineBackup.value,
        "DeviceProtection": data.DeviceProtection.value,
        "TechSupport": data.TechSupport.value,
        "StreamingTV": data.StreamingTV.value,
        "StreamingMovies": data.StreamingMovies.value,
        "Contract": data.Contract.value,
        "PaperlessBilling": data.PaperlessBilling.value,
        "PaymentMethod": data.PaymentMethod.value,
    }

    df = pd.DataFrame([input_dict], columns=ALL_FEATURES)

    try:
        proba = model.predict_proba(df)[0]
        churn_prob = float(proba[1])
        prediction = int(churn_prob >= 0.5)
        confidence = float(max(proba))

        # Risk classification
        if churn_prob < 0.3:
            risk_level = "Low"
        elif churn_prob < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return PredictionResponse(
            prediction="Churn" if prediction == 1 else "No Churn",
            churn_probability=round(churn_prob, 4),
            risk_level=risk_level,
            confidence=round(confidence, 4),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ---------------------------------------------------------------------------
# Embedded HTML Frontend — served at root
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def serve_frontend():
    """Serve the embedded web UI for churn prediction."""
    return FRONTEND_HTML


# ---------------------------------------------------------------------------
# HTML Frontend Template
# ---------------------------------------------------------------------------
FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Churn AI</title>
    <meta name="description" content="Minimalist customer churn prediction interface.">
    <!-- Using Inter for a very clean, modern geometric sans-serif look -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        /* CSS Reset & Base */
        *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --bg-body: #fafafa;
            --bg-card: #ffffff;
            --text-main: #111827;
            --text-muted: #6b7280;
            --text-placeholder: #9ca3af;
            --border: #e5e7eb;
            --border-focus: #111827;
            --accent: #111827; /* Monochrome accent */
            --accent-hover: #374151;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
            --radius: 8px;
            --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-body);
            color: var(--text-main);
            min-height: 100vh;
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 4rem 1.5rem;
        }

        /* Typography & Header */
        .header {
            margin-bottom: 3.5rem;
            animation: slideDown 0.6s ease-out;
        }

        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        h1 {
            font-size: 2.25rem;
            font-weight: 600;
            letter-spacing: -0.03em;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            color: var(--text-muted);
            font-weight: 300;
            font-size: 1.125rem;
            letter-spacing: -0.01em;
        }

        /* Health Badge */
        .health-status {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 1rem;
            font-size: 0.875rem;
            color: var(--text-muted);
            background: var(--bg-card);
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            border: 1px solid var(--border);
            box-shadow: var(--shadow-sm);
        }

        .dot {
            width: 6px; height: 6px;
            border-radius: 50%;
            background: var(--text-placeholder);
        }
        .dot.healthy { background: var(--success); }
        .dot.error { background: var(--danger); }

        /* Form Structure */
        .section-wrapper {
            margin-bottom: 2.5rem;
            animation: fadeIn 0.8s ease-out backwards;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .section-title {
            font-size: 0.875rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }

        .grid {
            display: grid;
            gap: 1.5rem 1rem;
        }
        .grid-2 { grid-template-columns: repeat(2, 1fr); }
        .grid-3 { grid-template-columns: repeat(3, 1fr); }
        
        @media (max-width: 640px) {
            .grid-2, .grid-3 { grid-template-columns: 1fr; }
            .container { padding: 2rem 1rem; }
        }

        /* Inputs */
        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.375rem;
        }

        label {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-main);
        }

        select, input[type="number"] {
            width: 100%;
            padding: 0.75rem;
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            font-size: 0.875rem;
            transition: var(--transition);
            box-shadow: var(--shadow-sm);
            outline: none;
            -webkit-appearance: none;
            appearance: none;
        }

        select {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%236b7280'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 0.75rem center;
            background-size: 1rem;
            padding-right: 2.5rem;
        }

        select:hover, input[type="number"]:hover {
            border-color: #d1d5db;
        }

        select:focus, input[type="number"]:focus {
            border-color: var(--border-focus);
            box-shadow: 0 0 0 1px var(--border-focus);
        }

        /* Button */
        .btn-submit {
            display: inline-flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            padding: 0.875rem;
            background-color: var(--accent);
            color: #ffffff;
            border: none;
            border-radius: var(--radius);
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            transition: var(--transition);
            margin-top: 1.5rem;
        }

        .btn-submit:hover {
            background-color: var(--accent-hover);
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        .btn-submit:disabled {
            opacity: 0.7;
            cursor: not-allowed;
            transform: none;
        }

        /* Loader */
        .spinner {
            display: none;
            width: 1.25rem;
            height: 1.25rem;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: #ffffff;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin { 100% { transform: rotate(360deg); } }

        .btn-submit.loading .btn-text { display: none; }
        .btn-submit.loading .spinner { display: block; }

        /* Results Area */
        .result-container {
            margin-top: 3rem;
            padding: 2.5rem;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow-lg);
            display: none;
            opacity: 0;
            transform: translateY(10px);
            transition: var(--transition);
        }

        .result-container.visible {
            display: block;
            opacity: 1;
            transform: translateY(0);
        }

        .result-header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .prediction-value {
            font-size: 2.5rem;
            font-weight: 600;
            letter-spacing: -0.05em;
            margin-bottom: 0.5rem;
        }

        .prediction-value.churn { color: var(--danger); }
        .prediction-value.stay { color: var(--success); }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
            text-align: center;
            border-top: 1px solid var(--border);
            padding-top: 2rem;
        }

        .stat-item {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .stat-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .stat-value {
            font-size: 1.25rem;
            font-weight: 500;
            color: var(--text-main);
        }
        
        .stat-value.high-risk { color: var(--danger); }
        .stat-value.med-risk { color: var(--warning); }
        .stat-value.low-risk { color: var(--success); }

        /* Progress Bar */
        .progress-bg {
            width: 100%;
            height: 4px;
            background: var(--border);
            border-radius: 2px;
            margin-top: 0.5rem;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: var(--accent);
            border-radius: 2px;
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Error */
        .error-message {
            margin-top: 1.5rem;
            padding: 1rem;
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #b91c1c;
            border-radius: var(--radius);
            font-size: 0.875rem;
            display: none;
        }
        .error-message.visible { display: block; }

        /* Footer */
        .footer {
            margin-top: 4rem;
            text-align: center;
            font-size: 0.75rem;
            color: var(--text-placeholder);
        }
        .footer a {
            color: var(--text-muted);
            text-decoration: none;
            transition: var(--transition);
        }
        .footer a:hover { color: var(--text-main); }

    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Churn AI.</h1>
        <p class="subtitle">Predictive analytics for customer retention.</p>
        <div class="health-status" id="healthBadge">
            <span class="dot" id="healthDot"></span>
            <span id="healthText">Checking API status...</span>
        </div>
    </div>

    <form id="predictionForm" onsubmit="return handlePrediction(event)">
        
        <!-- Demographics -->
        <div class="section-wrapper" style="animation-delay: 0.1s;">
            <div class="section-title">Demographics</div>
            <div class="grid grid-3">
                <div class="input-group">
                    <label for="gender">Gender</label>
                    <select id="gender" required>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="SeniorCitizen">Senior Citizen</label>
                    <select id="SeniorCitizen" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="Partner">Partner</label>
                    <select id="Partner" required>
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="Dependents">Dependents</label>
                    <select id="Dependents" required>
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                    </select>
                </div>
            </div>
        </div>

        <!-- Account -->
        <div class="section-wrapper" style="animation-delay: 0.2s;">
            <div class="section-title">Account Details</div>
            <div class="grid grid-3">
                <div class="input-group">
                    <label for="tenure">Tenure (Months)</label>
                    <input type="number" id="tenure" value="12" min="0" max="72" required>
                </div>
                <div class="input-group">
                    <label for="MonthlyCharges">Monthly Charges ($)</label>
                    <input type="number" id="MonthlyCharges" value="70.35" min="0" max="200" step="0.01" required>
                </div>
                <div class="input-group">
                    <label for="TotalCharges">Total Charges ($)</label>
                    <input type="number" id="TotalCharges" value="844.20" min="0" step="0.01" required>
                </div>
                <div class="input-group">
                    <label for="Contract">Contract Type</label>
                    <select id="Contract" required>
                        <option value="Month-to-month">Month-to-month</option>
                        <option value="One year">One year</option>
                        <option value="Two year">Two year</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="PaperlessBilling">Paperless Billing</label>
                    <select id="PaperlessBilling" required>
                        <option value="Yes">Yes</option>
                        <option value="No">No</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="PaymentMethod">Payment Method</label>
                    <select id="PaymentMethod" required>
                        <option value="Electronic check">Electronic check</option>
                        <option value="Mailed check">Mailed check</option>
                        <option value="Bank transfer (automatic)">Bank transfer (auto)</option>
                        <option value="Credit card (automatic)">Credit card (auto)</option>
                    </select>
                </div>
            </div>
        </div>

        <!-- Services -->
        <div class="section-wrapper" style="animation-delay: 0.3s;">
            <div class="section-title">Services</div>
            <div class="grid grid-3">
                <div class="input-group">
                    <label for="PhoneService">Phone Service</label>
                    <select id="PhoneService" required>
                        <option value="Yes">Yes</option>
                        <option value="No">No</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="MultipleLines">Multiple Lines</label>
                    <select id="MultipleLines" required>
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                        <option value="No phone service">No phone service</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="InternetService">Internet Service</label>
                    <select id="InternetService" required>
                        <option value="Fiber optic">Fiber optic</option>
                        <option value="DSL">DSL</option>
                        <option value="No">No</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="OnlineSecurity">Online Security</label>
                    <select id="OnlineSecurity" required>
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                        <option value="No internet service">No internet service</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="OnlineBackup">Online Backup</label>
                    <select id="OnlineBackup" required>
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                        <option value="No internet service">No internet service</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="DeviceProtection">Device Protection</label>
                    <select id="DeviceProtection" required>
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                        <option value="No internet service">No internet service</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="TechSupport">Tech Support</label>
                    <select id="TechSupport" required>
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                        <option value="No internet service">No internet service</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="StreamingTV">Streaming TV</label>
                    <select id="StreamingTV" required>
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                        <option value="No internet service">No internet service</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="StreamingMovies">Streaming Movies</label>
                    <select id="StreamingMovies" required>
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                        <option value="No internet service">No internet service</option>
                    </select>
                </div>
            </div>
        </div>

        <button type="submit" class="btn-submit" id="submitBtn">
            <span class="btn-text">Generate Prediction</span>
            <div class="spinner"></div>
        </button>
    </form>

    <div class="error-message" id="errorBox"></div>

    <div class="result-container" id="resultBox">
        <div class="result-header">
            <div class="stat-label" style="margin-bottom: 0.5rem;">Predicted Outcome</div>
            <div class="prediction-value" id="predValue">--</div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-item">
                <span class="stat-label">Probability</span>
                <span class="stat-value" id="predProb">--</span>
                <div class="progress-bg">
                    <div class="progress-fill" id="probBar" style="width: 0%;"></div>
                </div>
            </div>
            <div class="stat-item">
                <span class="stat-label">Risk Level</span>
                <span class="stat-value" id="predRisk">--</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Model Confidence</span>
                <span class="stat-value" id="predConf">--</span>
            </div>
        </div>
    </div>

    <div class="footer">
        <p><a href="/docs" target="_blank">API Reference</a> &nbsp;&middot;&nbsp; <a href="/health" target="_blank">System Status</a></p>
    </div>
</div>

<script>
    // System Health Check
    async function checkSystem() {
        const dot = document.getElementById('healthDot');
        const text = document.getElementById('healthText');
        try {
            const res = await fetch('/health');
            const data = await res.json();
            if (data.model_loaded) {
                dot.className = 'dot healthy';
                text.textContent = 'System Operational';
            } else {
                dot.className = 'dot error';
                text.textContent = 'Model Not Loaded';
            }
        } catch {
            dot.className = 'dot error';
            text.textContent = 'API Offline';
        }
    }
    checkSystem();

    // Field Mapping
    const fields = [
        'gender','SeniorCitizen','Partner','Dependents','tenure',
        'PhoneService','MultipleLines','InternetService',
        'OnlineSecurity','OnlineBackup','DeviceProtection',
        'TechSupport','StreamingTV','StreamingMovies',
        'Contract','PaperlessBilling','PaymentMethod',
        'MonthlyCharges','TotalCharges'
    ];
    const numeric = ['tenure', 'MonthlyCharges', 'TotalCharges'];

    // Auto-calculations
    function updateTotal() {
        const t = parseFloat(document.getElementById('tenure').value) || 0;
        const m = parseFloat(document.getElementById('MonthlyCharges').value) || 0;
        document.getElementById('TotalCharges').value = (t * m).toFixed(2);
    }
    document.getElementById('tenure').addEventListener('input', updateTotal);
    document.getElementById('MonthlyCharges').addEventListener('input', updateTotal);

    document.getElementById('PaperlessBilling').addEventListener('change', (e) => {
        if (e.target.value === 'No') {
            document.getElementById('PaymentMethod').value = 'Mailed check';
        }
    });

    // Handle Submission
    async function handlePrediction(e) {
        e.preventDefault();
        
        const btn = document.getElementById('submitBtn');
        const errBox = document.getElementById('errorBox');
        const resBox = document.getElementById('resultBox');
        
        btn.classList.add('loading');
        btn.disabled = true;
        errBox.classList.remove('visible');
        resBox.classList.remove('visible');
        
        // Reset progress bar for animation
        document.getElementById('probBar').style.width = '0%';

        const payload = {};
        for (const f of fields) {
            const val = document.getElementById(f).value;
            payload[f] = numeric.includes(f) ? parseFloat(val) : val;
        }

        try {
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Prediction failed');
            }

            const data = await res.json();
            renderResult(data);
        } catch (err) {
            errBox.textContent = err.message;
            errBox.classList.add('visible');
        } finally {
            btn.classList.remove('loading');
            btn.disabled = false;
        }
    }

    // Render Result
    function renderResult(data) {
        const resBox = document.getElementById('resultBox');
        
        // Main Prediction
        const valEl = document.getElementById('predValue');
        valEl.textContent = data.prediction;
        valEl.className = 'prediction-value ' + (data.prediction === 'Churn' ? 'churn' : 'stay');

        // Probability
        const probPct = (data.churn_probability * 100).toFixed(1) + '%';
        document.getElementById('predProb').textContent = probPct;
        
        // Bar Animation (slight delay for effect)
        setTimeout(() => {
            const bar = document.getElementById('probBar');
            bar.style.width = probPct;
            if (data.churn_probability < 0.3) {
                bar.style.backgroundColor = 'var(--success)';
            } else if (data.churn_probability < 0.6) {
                bar.style.backgroundColor = 'var(--warning)';
            } else {
                bar.style.backgroundColor = 'var(--danger)';
            }
        }, 100);

        // Risk Level
        const riskEl = document.getElementById('predRisk');
        riskEl.textContent = data.risk_level;
        
        let riskClass = 'stat-value ';
        if(data.risk_level === 'High') riskClass += 'high-risk';
        else if(data.risk_level === 'Medium') riskClass += 'med-risk';
        else riskClass += 'low-risk';
        riskEl.className = riskClass;

        // Confidence
        document.getElementById('predConf').textContent = (data.confidence * 100).toFixed(1) + '%';

        resBox.classList.add('visible');
        resBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
</script>
</body>
</html>"""
