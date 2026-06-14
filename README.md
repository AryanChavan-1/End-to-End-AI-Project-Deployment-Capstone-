# 🚀 End-to-End AI Project Deployment — Customer Churn Predictor

A production-grade machine learning application that predicts customer churn using a Scikit-Learn Random Forest pipeline, served via FastAPI with an embedded web UI, containerized with Docker, and deployed to the cloud.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-orange?logo=scikit-learn)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [Docker](#docker)
- [Cloud Deployment (Render)](#cloud-deployment-render)
- [Testing](#testing)

---

## 🏗️ Architecture

```
┌─────────────────┐     HTTP POST     ┌──────────────────────────┐
│   Web Browser   │ ───────────────── │    FastAPI Backend        │
│  (Embedded UI)  │                   │  ┌────────────────────┐  │
│  or Streamlit   │ ◄─── JSON ─────  │  │ Scikit-Learn        │  │
│                 │                   │  │ Pipeline (.pkl)     │  │
└─────────────────┘                   │  │ Preprocessor+Model  │  │
                                      │  └────────────────────┘  │
                                      └──────────────────────────┘
```

**ML Pipeline**: `ColumnTransformer (StandardScaler + OneHotEncoder) → RandomForestClassifier`

All preprocessing and classification are bundled into a **single serialized pipeline** — no separate preprocessor files needed.

---

## 📁 Project Structure

```
├── model/
│   ├── train.py               # Training script
│   ├── churn_pipeline.pkl     # Serialized ML pipeline
│   └── feature_names.pkl      # Feature column order
├── app/
│   ├── __init__.py
│   └── main.py                # FastAPI application (API + embedded frontend)
├── frontend/
│   └── streamlit_app.py       # Streamlit dashboard (alternative UI)
├── data/
│   └── Telco-Customer-Churn.csv  # Dataset (auto-generated if missing)
├── tests/
│   └── test_api.py            # Pytest test suite
├── Dockerfile                 # Container build instructions
├── render.yaml                # Render deployment blueprint
├── requirements.txt           # Python dependencies
├── .gitignore
├── .dockerignore
└── README.md                  # This file
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/End-to-End-AI-Project-Deployment-Capstone-.git
cd End-to-End-AI-Project-Deployment-Capstone-
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python model/train.py
```

This will:
- Generate synthetic Telco Customer Churn data (or use `data/Telco-Customer-Churn.csv` if present)
- Train a Random Forest pipeline
- Save `model/churn_pipeline.pkl` and `model/feature_names.pkl`
- Print the classification report and accuracy

### 4. Start the API Server

```bash
uvicorn app.main:app --reload
```

The API is now live at **http://localhost:8000**

- 🌐 **Web UI**: http://localhost:8000
- 📖 **Swagger Docs**: http://localhost:8000/docs
- ❤️ **Health Check**: http://localhost:8000/health

### 5. (Optional) Start Streamlit Frontend

In a second terminal:

```bash
streamlit run frontend/streamlit_app.py
```

The Streamlit dashboard is now live at **http://localhost:8501**

---

## 📡 API Reference

### `GET /health`

Health check endpoint.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

### `POST /predict`

Predict customer churn.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "StreamingMovies": "No"
  }'
```

**Response:**
```json
{
  "prediction": "Churn",
  "churn_probability": 0.7350,
  "risk_level": "High",
  "confidence": 0.7350
}
```

### Input Fields

| Field | Type | Valid Values |
|-------|------|-------------|
| `gender` | string | `Male`, `Female` |
| `SeniorCitizen` | string | `0`, `1` |
| `Partner` | string | `Yes`, `No` |
| `Dependents` | string | `Yes`, `No` |
| `tenure` | integer | 0–72 |
| `Contract` | string | `Month-to-month`, `One year`, `Two year` |
| `PaperlessBilling` | string | `Yes`, `No` |
| `PaymentMethod` | string | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| `MonthlyCharges` | float | 0–200 |
| `TotalCharges` | float | ≥ 0 |
| `PhoneService` | string | `Yes`, `No` |
| `MultipleLines` | string | `Yes`, `No`, `No phone service` |
| `InternetService` | string | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity` | string | `Yes`, `No`, `No internet service` |
| `OnlineBackup` | string | `Yes`, `No`, `No internet service` |
| `DeviceProtection` | string | `Yes`, `No`, `No internet service` |
| `TechSupport` | string | `Yes`, `No`, `No internet service` |
| `StreamingTV` | string | `Yes`, `No`, `No internet service` |
| `StreamingMovies` | string | `Yes`, `No`, `No internet service` |

---

## 🐳 Docker

### Build the Image

```bash
docker build -t churn-predictor .
```

### Run the Container

```bash
docker run -p 8000:8000 churn-predictor
```

The app is now running at **http://localhost:8000**

> The model is trained during the Docker build step, so the container is self-contained.

---

## ☁️ Cloud Deployment (Render)

### One-Click Deploy

1. Push this repo to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click **New → Blueprint**
4. Connect your GitHub repo
5. Render will auto-detect `render.yaml` and deploy

### Manual Deploy

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **New → Web Service**
3. Connect your GitHub repo
4. Settings:
   - **Runtime**: Docker
   - **Plan**: Free
   - **Health Check Path**: `/health`
5. Click **Deploy**

Your app will be live at: `https://your-app-name.onrender.com`

> **Note**: Free tier services spin down after 15 min of inactivity. First request after sleep takes ~30s (cold start).

---

## 🧪 Testing

### Run All Tests

```bash
pip install pytest
pytest tests/test_api.py -v
```

### Test with Postman

1. Import the API URL: `http://localhost:8000`
2. `GET /health` → Should return status 200
3. `POST /predict` → Use the JSON payload from the API Reference above

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Model** | Scikit-Learn (RandomForestClassifier) |
| **API Framework** | FastAPI + Uvicorn |
| **Validation** | Pydantic v2 |
| **Frontend** | Embedded HTML/CSS/JS + Streamlit |
| **Containerization** | Docker |
| **Deployment** | Render (free tier) |

---

## 📝 License

This project is part of the End-to-End AI Project Deployment Capstone.