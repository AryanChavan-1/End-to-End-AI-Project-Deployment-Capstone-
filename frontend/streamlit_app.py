"""
Customer Churn Predictor — Streamlit Frontend

A polished dashboard that connects to the FastAPI backend API
for interactive churn prediction.

Usage:
    streamlit run frontend/streamlit_app.py
"""

import os
import streamlit as st
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark premium theme
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0f172a 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(17, 24, 39, 0.95);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }

    /* Headers */
    h1 { color: #e2e8f0 !important; }
    h2, h3 { color: #cbd5e1 !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1rem;
    }

    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }

    /* Success/Error messages */
    .stSuccess { border-radius: 12px; }
    .stError { border-radius: 12px; }

    /* Form button */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.35) !important;
        transform: translateY(-2px) !important;
    }

    /* Divider */
    hr { border-color: rgba(99, 102, 241, 0.15) !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Health Check (cached for 60s)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60)
def check_api_health() -> dict:
    """Check if the backend API is available."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except (requests.ConnectionError, requests.Timeout):
        pass
    return None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# ⚡ Customer Churn Predictor")
st.markdown(
    "*AI-powered prediction using a Random Forest pipeline — "
    "Enter customer details to predict churn risk.*"
)

# Health status
health = check_api_health()
if health and health.get("model_loaded"):
    st.success(f"🟢 API Online — Model v{health.get('version', '?')} loaded")
elif health:
    st.warning("🟡 API Online — but model not loaded. Run `python model/train.py` first.")
else:
    st.error(
        f"🔴 Cannot connect to API at `{API_URL}`. "
        "Make sure the backend is running: `uvicorn app.main:app --reload`"
    )

st.divider()

# ---------------------------------------------------------------------------
# Input Form
# ---------------------------------------------------------------------------
with st.container():

    # --- Demographics ---
    st.subheader("👤 Demographics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col3:
        partner = st.selectbox("Partner", ["Yes", "No"])
    with col4:
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    st.divider()

    # --- Account ---
    st.subheader("📋 Account Information")
    
    if "tenure" not in st.session_state:
        st.session_state.tenure = 12
    if "monthly_charges" not in st.session_state:
        st.session_state.monthly_charges = 70.35
    if "total_charges" not in st.session_state:
        st.session_state.total_charges = 844.20
    if "paperless" not in st.session_state:
        st.session_state.paperless = "Yes"
    if "payment" not in st.session_state:
        st.session_state.payment = "Electronic check"

    def update_total():
        st.session_state.total_charges = st.session_state.tenure * st.session_state.monthly_charges

    def update_payment():
        if st.session_state.paperless == "No":
            st.session_state.payment = "Mailed check"

    col5, col6, col7 = st.columns(3)

    with col5:
        tenure = st.slider("Tenure (months)", 0, 72, key="tenure", on_change=update_total)
    with col6:
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, step=0.5, key="monthly_charges", on_change=update_total)
    with col7:
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, step=1.0, key="total_charges")

    col8, col9, col10 = st.columns(3)

    with col8:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with col9:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key="paperless", on_change=update_payment)
    with col10:
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            key="payment"
        )

    st.divider()

    # --- Services ---
    st.subheader("🔌 Services")
    col11, col12, col13 = st.columns(3)

    with col11:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

    with col12:
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    with col13:
        tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.divider()

    # Submit
    submitted = st.button("🔍 Predict Churn", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": str(senior),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
    }

    with st.spinner("⏳ Analyzing customer data..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()

                st.divider()
                st.subheader("📊 Prediction Results")

                # Metrics row
                m1, m2, m3, m4 = st.columns(4)

                with m1:
                    icon = "⚠️" if result["prediction"] == "Churn" else "✅"
                    st.metric("Prediction", f"{icon} {result['prediction']}")

                with m2:
                    prob_pct = f"{result['churn_probability'] * 100:.1f}%"
                    st.metric("Churn Probability", prob_pct)

                with m3:
                    risk = result["risk_level"]
                    risk_icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "⚪")
                    st.metric("Risk Level", f"{risk_icon} {risk}")

                with m4:
                    conf_pct = f"{result['confidence'] * 100:.1f}%"
                    st.metric("Confidence", conf_pct)

                # Progress bar for probability
                st.progress(result["churn_probability"])

                # Interpretation
                if result["prediction"] == "Churn":
                    st.error(
                        f"⚠️ **High risk of churn detected!** "
                        f"This customer has a {prob_pct} probability of leaving. "
                        f"Consider offering retention incentives like contract upgrades, "
                        f"discounts, or improved support."
                    )
                else:
                    st.success(
                        f"✅ **This customer is likely to stay.** "
                        f"Churn probability is only {prob_pct}. "
                        f"Continue providing excellent service to maintain loyalty."
                    )

            elif response.status_code == 422:
                errors = response.json().get("detail", [])
                st.error(f"❌ Validation Error: {errors}")

            elif response.status_code == 503:
                st.error("❌ Model not loaded. Please run `python model/train.py` first.")

            else:
                st.error(f"❌ API returned status {response.status_code}")

        except requests.Timeout:
            st.error(
                "⏰ Request timed out. The API might be starting up (cold start). "
                "Please try again in a few seconds."
            )
        except requests.ConnectionError:
            st.error(
                f"🔴 Cannot connect to `{API_URL}`. "
                "Make sure the backend is running: `uvicorn app.main:app --reload`"
            )

# ---------------------------------------------------------------------------
# Sidebar — Info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown(
        """
    This application predicts whether a telecom customer
    will **churn** (leave the service) based on their account
    information, demographics, and subscribed services.

    **Model**: Random Forest Classifier
    **Features**: 19 input features
    **Backend**: FastAPI + Scikit-Learn

    ---

    ### 🔗 Quick Links
    - [API Docs (Swagger)](""" + API_URL + """/docs)
    - [Health Check](""" + API_URL + """/health)

    ---

    ### 📝 How to Use
    1. Fill in all customer details
    2. Click **Predict Churn**
    3. Review the risk assessment

    ---

    *Built as part of the End-to-End AI
    Project Deployment Capstone.*
    """
    )
