"""
API Tests — Customer Churn Prediction

Tests for /health and /predict endpoints using FastAPI TestClient.

Usage:
    pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------------------------
class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_status_is_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"


# ---------------------------------------------------------------------------
# Predict Endpoint
# ---------------------------------------------------------------------------
VALID_PAYLOAD = {
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


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_predict_has_required_fields(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "prediction" in data
        assert "churn_probability" in data
        assert "risk_level" in data
        assert "confidence" in data

    def test_prediction_is_valid_value(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert data["prediction"] in ["Churn", "No Churn"]

    def test_probability_in_range(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_risk_level_is_valid(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert data["risk_level"] in ["Low", "Medium", "High"]

    def test_confidence_in_range(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_invalid_gender_returns_422(self, client):
        bad = {**VALID_PAYLOAD, "gender": "Unknown"}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_negative_tenure_returns_422(self, client):
        bad = {**VALID_PAYLOAD, "tenure": -5}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_tenure_exceeds_max_returns_422(self, client):
        bad = {**VALID_PAYLOAD, "tenure": 100}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_missing_field_returns_422(self, client):
        incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "gender"}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_low_risk_customer(self, client):
        """A two-year contract customer with full services should be low risk."""
        safe_customer = {
            **VALID_PAYLOAD,
            "Contract": "Two year",
            "tenure": 60,
            "TotalCharges": 4200.0,
            "OnlineSecurity": "Yes",
            "TechSupport": "Yes",
            "PaymentMethod": "Bank transfer (automatic)",
        }
        data = client.post("/predict", json=safe_customer).json()
        # Not asserting exact prediction since it depends on model,
        # but probability should be relatively low
        assert data["churn_probability"] < 0.8  # Reasonable upper bound


# ---------------------------------------------------------------------------
# Frontend Endpoint
# ---------------------------------------------------------------------------
class TestFrontendEndpoint:
    """Tests for GET / (embedded HTML frontend)."""

    def test_root_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_root_contains_form(self, client):
        response = client.get("/")
        assert "predictionForm" in response.text
        assert "Generate Prediction" in response.text
