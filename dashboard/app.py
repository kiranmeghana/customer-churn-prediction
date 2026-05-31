"""
Streamlit Dashboard: Customer Churn Prediction Monitoring
Run: streamlit run dashboard/app.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "https://customer-churn-prediction-5fgm.onrender.com"

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00d4ff;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8ab4d4;
        margin-top: 5px;
    }
    .risk-high { color: #ff4444; font-weight: bold; }
    .risk-medium { color: #ff9800; font-weight: bold; }
    .risk-low { color: #4caf50; font-weight: bold; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ───────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_data():
    """Load processed data for dashboard visualizations."""
    try:
        df = pd.read_csv("data/processed/features.csv")
        return df
    except FileNotFoundError:
        return None


@st.cache_data(ttl=30)
def load_metrics():
    """Load model metrics."""
    try:
        with open("models/metrics_summary.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def check_api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json()
    except Exception:
        return {"status": "offline", "model_loaded": False}


def predict_customer(payload: dict):
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
        if r.status_code == 200:
            return r.json(), None
        return None, r.json().get("detail", "Unknown error")
    except Exception as e:
        return None, str(e)


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/examples/data/streamlit-logo-primary-colormark-darktext.png", width=120)
    st.markdown("## 🔧 Navigation")
    page = st.selectbox(
        "Select Page",
        ["📊 Overview", "🔮 Live Prediction", "📈 Model Performance", "🧩 Feature Analysis", "🚨 Monitoring"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    health = check_api_health()
    status_color = "🟢" if health.get("status") == "healthy" else "🔴"
    st.markdown(f"**API Status:** {status_color} {health.get('status', 'unknown').upper()}")
    st.markdown(f"**Model Loaded:** {'✅' if health.get('model_loaded') else '❌'}")
    st.markdown("---")
    st.markdown("### Quick Links")
    st.markdown(f"[📖 API Docs]({API_BASE}/docs)")
    st.markdown(f"[📊 MLflow UI](http://localhost:5000)")
    st.markdown("---")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()


# ── Page: Overview ─────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 Customer Churn Prediction")
    st.markdown("### End-to-End ML Pipeline Monitoring Dashboard")

    df = load_data()
    metrics = load_metrics()

    if df is None:
        st.error("⚠️ Data not found. Run the pipeline first: `python scripts/run_pipeline.py`")
        st.code("python scripts/run_pipeline.py", language="bash")
        st.stop()

    # KPI Row
    total = len(df)
    churn_rate = df["churn"].mean()
    churned = df["churn"].sum()
    retained = total - churned

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{total:,}")
    with col2:
        st.metric("Churned", f"{churned:,}", delta=f"{churn_rate:.1%} rate", delta_color="inverse")
    with col3:
        st.metric("Retained", f"{retained:,}")
    with col4:
        if metrics:
            best_auc = metrics.get("best_roc_auc", 0)
            st.metric("Best Model AUC", f"{best_auc:.4f}")
        else:
            st.metric("Best Model AUC", "Run pipeline")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        churn_counts = df["churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        churn_counts["Label"] = churn_counts["Churn"].map({0: "Retained", 1: "Churned"})
        fig = px.pie(
            churn_counts, values="Count", names="Label",
            color_discrete_sequence=["#00d4ff", "#ff4444"],
            hole=0.4,
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn by Contract Type")
        churn_by_contract = df.groupby("contract_type")["churn"].mean().reset_index()
        churn_by_contract.columns = ["Contract", "Churn Rate"]
        fig = px.bar(
            churn_by_contract, x="Contract", y="Churn Rate",
            color="Churn Rate",
            color_continuous_scale=["#00d4ff", "#ff4444"],
            text_auto=".1%",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Charges vs Tenure")
        sample = df.sample(min(1000, len(df)), random_state=42)
        fig = px.scatter(
            sample, x="tenure", y="monthly_charges",
            color=sample["churn"].map({0: "Retained", 1: "Churned"}),
            color_discrete_map={"Retained": "#00d4ff", "Churned": "#ff4444"},
            opacity=0.6, size_max=5,
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn by Internet Service")
        churn_by_isp = df.groupby("internet_service")["churn"].mean().reset_index()
        churn_by_isp.columns = ["Internet Service", "Churn Rate"]
        fig = px.bar(
            churn_by_isp, x="Internet Service", y="Churn Rate",
            color="Churn Rate",
            color_continuous_scale=["#00d4ff", "#ff4444"],
            text_auto=".1%",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


# ── Page: Live Prediction ───────────────────────────────────────────────────
elif page == "🔮 Live Prediction":
    st.title("🔮 Live Churn Prediction")
    st.markdown("Enter customer details to get real-time churn prediction.")

    if not health.get("model_loaded"):
        st.error("⚠️ Model not loaded. Run the training pipeline first.")
        st.code("python scripts/run_pipeline.py", language="bash")

    with st.form("prediction_form"):
        st.subheader("Customer Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            customer_id = st.text_input("Customer ID", value="CUST_TEST_001")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.5)
            total_charges = st.number_input("Total Charges ($)", 0.0, 15000.0, float(tenure * monthly_charges))

        with col2:
            contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

        with col3:
            num_products = st.slider("Number of Products", 1, 5, 2)
            num_support_tickets = st.slider("Support Tickets", 0, 10, 1)
            avg_call_duration = st.number_input("Avg Call Duration (min)", 0.0, 60.0, 8.0)
            days_since_last_contact = st.slider("Days Since Last Contact", 0, 180, 30)

        st.subheader("Services")
        col1, col2, col3, col4, col5 = st.columns(5)
        has_tech_support = col1.checkbox("Tech Support")
        has_streaming = col2.checkbox("Streaming")
        has_online_security = col3.checkbox("Online Security")
        has_online_backup = col4.checkbox("Online Backup")
        paperless_billing = col5.checkbox("Paperless Billing", value=True)

        submitted = st.form_submit_button("🔮 Predict Churn", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "customer_id": customer_id,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "num_products": num_products,
            "has_tech_support": has_tech_support,
            "has_streaming": has_streaming,
            "has_online_security": has_online_security,
            "has_online_backup": has_online_backup,
            "paperless_billing": paperless_billing,
            "num_support_tickets": num_support_tickets,
            "avg_call_duration": avg_call_duration,
            "days_since_last_contact": days_since_last_contact,
        }

        with st.spinner("Predicting..."):
            result, error = predict_customer(payload)

        if error:
            st.error(f"❌ Prediction failed: {error}")
            st.info("Make sure the API is running: `uvicorn src.serving.api:app --reload --port 8000`")
        else:
            prob = result["churn_probability"]
            risk = result["risk_level"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Churn Probability", f"{prob:.1%}")
            col2.metric("Prediction", "🚨 CHURN" if result["churn_prediction"] else "✅ RETAIN")
            col3.metric("Risk Level", risk)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Churn Risk Score", "font": {"size": 20}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ff4444" if prob > 0.5 else "#00d4ff"},
                    "steps": [
                        {"range": [0, 25], "color": "#1a4a2e"},
                        {"range": [25, 50], "color": "#2d4a1e"},
                        {"range": [50, 75], "color": "#4a3a1a"},
                        {"range": [75, 100], "color": "#4a1a1a"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
                number={"suffix": "%", "font": {"size": 30}},
            ))
            fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)


# ── Page: Model Performance ─────────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    metrics = load_metrics()
    if metrics is None:
        st.warning("No metrics found. Run: `python scripts/run_pipeline.py`")
        st.stop()

    st.subheader(f"Best Model: **{metrics.get('best_model', 'N/A')}**")
    st.metric("Best ROC-AUC", f"{metrics.get('best_roc_auc', 0):.4f}")

    all_results = metrics.get("all_results", {})
    if all_results:
        rows = []
        for model_name, m in all_results.items():
            rows.append({"Model": model_name, **m})
        df_metrics = pd.DataFrame(rows)

        st.subheader("All Models Comparison")
        st.dataframe(df_metrics.style.highlight_max(
            subset=["roc_auc", "accuracy", "f1"],
            color="#1a4a2e"
        ), use_container_width=True)

        fig = px.bar(
            df_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Metric", y="Score", color="Model", barmode="group",
            color_discrete_sequence=["#00d4ff", "#ff9800"],
            title="Model Metrics Comparison",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    # Show saved plots
    plots_dir = Path("models/plots")
    if plots_dir.exists():
        plot_files = list(plots_dir.glob("*.png"))
        if plot_files:
            st.subheader("Training Plots")
            cols = st.columns(min(2, len(plot_files)))
            for i, plot_path in enumerate(plot_files):
                cols[i % 2].image(str(plot_path), caption=plot_path.stem.replace("_", " ").title())


# ── Page: Feature Analysis ──────────────────────────────────────────────────
elif page == "🧩 Feature Analysis":
    st.title("🧩 Feature Analysis")

    df = load_data()
    if df is None:
        st.error("Data not found.")
        st.stop()

    st.subheader("Feature Distributions")
    numeric_cols = ["tenure", "monthly_charges", "total_charges",
                    "num_products", "num_support_tickets", "avg_call_duration"]

    selected_col = st.selectbox("Select Feature", numeric_cols)
    fig = px.histogram(
        df, x=selected_col, color=df["churn"].map({0: "Retained", 1: "Churned"}),
        barmode="overlay", opacity=0.7,
        color_discrete_map={"Retained": "#00d4ff", "Churned": "#ff4444"},
        title=f"Distribution of {selected_col} by Churn Status",
        nbins=50,
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation with Churn")
    bool_cols = ["has_tech_support", "has_streaming", "has_online_security",
                 "has_online_backup", "paperless_billing"]
    all_numeric = numeric_cols + bool_cols

    existing_cols = [c for c in all_numeric if c in df.columns]
    corr_df = df[existing_cols + ["churn"]].copy()
    for col in bool_cols:
        if col in corr_df.columns:
            corr_df[col] = corr_df[col].astype(int)

    correlations = corr_df.corr()["churn"].drop("churn").sort_values(ascending=False)
    fig = px.bar(
        x=correlations.index, y=correlations.values,
        color=correlations.values,
        color_continuous_scale=["#00d4ff", "white", "#ff4444"],
        title="Feature Correlation with Churn",
        labels={"x": "Feature", "y": "Correlation"},
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Churn Rates by Category")
    cat_col = st.selectbox("Categorical Feature", ["contract_type", "payment_method", "internet_service", "gender", "senior_citizen"])
    cat_churn = df.groupby(cat_col)["churn"].agg(["mean", "count"]).reset_index()
    cat_churn.columns = [cat_col, "Churn Rate", "Count"]
    fig = px.bar(
        cat_churn, x=cat_col, y="Churn Rate",
        text_auto=".1%",
        color="Churn Rate",
        color_continuous_scale=["#00d4ff", "#ff4444"],
        title=f"Churn Rate by {cat_col}",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# ── Page: Monitoring ────────────────────────────────────────────────────────
elif page == "🚨 Monitoring":
    st.title("🚨 Model Monitoring")

    df = load_data()
    if df is None:
        st.error("Data not found.")
        st.stop()

    st.subheader("Data Drift Simulation")
    st.info("Simulating drift detection between reference (70%) and current (30%) data splits.")

    if st.button("🔍 Run Drift Detection"):
        sys.path.insert(0, ".")
        try:
            from src.monitoring.monitor import DataDriftDetector
            reference = df.sample(frac=0.7, random_state=42)
            current = df.sample(frac=0.3, random_state=99)

            numeric_cols = ["tenure", "monthly_charges", "total_charges", "num_products"]
            detector = DataDriftDetector(reference[numeric_cols], threshold=0.1)
            report = detector.detect_drift(current[numeric_cols])

            col1, col2, col3 = st.columns(3)
            col1.metric("Features Analyzed", len(report["feature_drift"]))
            col2.metric("Drifted Features", len(report["drifted_features"]))
            col3.metric("Drift %", f"{report['drift_percentage']}%")

            if report["overall_drift"]:
                st.warning(f"⚠️ Drift detected in: {report['drifted_features']}")
            else:
                st.success("✅ No significant drift detected")

            # Feature drift table
            drift_rows = []
            for feat, info in report["feature_drift"].items():
                drift_rows.append({
                    "Feature": feat,
                    "Test": info["type"],
                    "Statistic": info["statistic"],
                    "Drift Detected": "🔴 YES" if info["drift_detected"] else "🟢 NO",
                })
            st.dataframe(pd.DataFrame(drift_rows), use_container_width=True)
        except ImportError as e:
            st.error(f"Import error: {e}")

    st.subheader("Recent Predictions Log")
    try:
        r = requests.get(f"{API_BASE}/predictions/recent?limit=20", timeout=3)
        if r.status_code == 200:
            data = r.json()
            st.metric("Total Predictions Made", data.get("total_predictions", 0))
            recent = data.get("recent", [])
            if recent:
                st.dataframe(pd.DataFrame(recent), use_container_width=True)
            else:
                st.info("No predictions yet. Use the Live Prediction page.")
        else:
            st.warning("Could not fetch prediction log from API.")
    except Exception:
        st.warning("API offline. Start with: `uvicorn src.serving.api:app --reload`")

    st.subheader("System Status")
    col1, col2 = st.columns(2)
    with col1:
        api_health = check_api_health()
        st.json(api_health)
    with col2:
        model_files = list(Path("models").glob("*.pkl")) if Path("models").exists() else []
        st.markdown("**Model Artifacts:**")
        for f in model_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            st.markdown(f"- `{f.name}` ({size_mb:.1f} MB)")
        if not model_files:
            st.warning("No model artifacts found.")
