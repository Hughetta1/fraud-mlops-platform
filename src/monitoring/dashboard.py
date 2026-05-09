"""
Real-Time Fraud Detection Dashboard

Interactive Streamlit dashboard for monitoring fraud detection system:
- Real-time transaction monitoring
- Model performance analytics
- Risk level distributions
- Feature importance visualization
- System health metrics
- Historical trend analysis

Author: Hughetta1
"""

import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Configure Streamlit page
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="[Fraud Detection]",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .metric-red {
        border-left-color: #ff4b4b !important;
    }
    .metric-orange {
        border-left-color: #ff8c00 !important;
    }
    .metric-green {
        border-left-color: #00cc00 !important;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# API Configuration
# Read API URL from Streamlit Secrets (cloud) or env var (Docker) or default (local)
API_BASE_URL = (
    st.secrets.get("API_BASE_URL") or
    os.getenv("API_BASE_URL") or
    "http://localhost:8000"
)


class TransactionSimulator:
    """Simulate real-time transaction data for dashboard demo"""

    def __init__(self):
        self.transaction_id_counter = 1000

    def generate_transaction(self, fraud_bias=0.1):
        """Generate a realistic transaction with controllable fraud probability"""

        # Generate V features — model expects V1-V28
        if random.random() < fraud_bias:
            # Fraud pattern: wider variance, some extreme values
            v_features = {f"V{i}": round(random.gauss(0, 3), 6) for i in range(1, 29)}
            amount = round(np.random.lognormal(4.5, 1.5), 2)
        else:
            # Normal pattern: tight around zero
            v_features = {f"V{i}": round(random.gauss(0, 1), 6) for i in range(1, 29)}
            amount = round(np.random.lognormal(3.2, 0.8), 2)

        # Generate transaction
        transaction = {
            "transaction_id": f"TXN_{self.transaction_id_counter}",
            "Time": time.time() % (24 * 3600),  # Time of day in seconds
            "Amount": round(max(0.01, amount), 2),
            **v_features,
        }

        self.transaction_id_counter += 1
        return transaction


@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_api_health():
    """Check API health status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json(), True
        else:
            return {
                "status": "unhealthy",
                "error": f"HTTP {response.status_code}",
            }, False
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}, False


@st.cache_data(ttl=10)  # Cache for 10 seconds
def get_api_metrics():
    """Get API performance metrics"""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=5)
        if response.status_code == 200:
            return response.json(), True
        else:
            return {}, False
    except Exception:
        return {}, False


def predict_transaction(transaction):
    """Send transaction to API for prediction"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict", json=transaction, timeout=10
        )
        if response.status_code == 200:
            return response.json(), True
        else:
            return {"error": f"HTTP {response.status_code}"}, False
    except Exception as e:
        return {"error": str(e)}, False


def create_gauge_chart(value, title, max_value=1, color_thresholds=None):
    """Create a gauge chart for metrics"""
    if color_thresholds is None:
        color_thresholds = [0.3, 0.7]

    # Determine color based on value
    if value <= color_thresholds[0]:
        color = "green"
    elif value <= color_thresholds[1]:
        color = "orange"
    else:
        color = "red"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title},
            delta={"reference": color_thresholds[0]},
            gauge={
                "axis": {"range": [None, max_value]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, color_thresholds[0]], "color": "lightgray"},
                    {
                        "range": [color_thresholds[0], color_thresholds[1]],
                        "color": "gray",
                    },
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": color_thresholds[1],
                },
            },
        )
    )

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig


@st.cache_data(ttl=30)
def get_monitoring_versions():
    """Get version manifest from API."""
    try:
        r = requests.get(f"{API_BASE_URL}/monitoring/versions", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


@st.cache_data(ttl=30)
def get_monitoring_report():
    """Get model monitoring report from API."""
    try:
        r = requests.get(f"{API_BASE_URL}/monitoring/report", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def render_model_health(metrics_data, metrics_available):
    """Render the Model Health dashboard page."""
    st.header("Model Health & Monitoring")

    # --- Row 1: Version info cards ---
    versions = get_monitoring_versions()
    report = get_monitoring_report()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        prod_ver = versions.get("production", "unknown")
        st.metric("Production Model", prod_ver if prod_ver else "none")
    with col2:
        shadow_ver = versions.get("shadow")
        st.metric("Shadow Model", shadow_ver if shadow_ver else "none")
    with col3:
        n_versions = len(versions.get("versions", {}))
        st.metric("Total Versions", n_versions)
    with col4:
        uptime = metrics_data.get("uptime_seconds", 0) if metrics_available else 0
        st.metric("API Uptime", f"{uptime/3600:.1f} hrs")

    st.divider()

    # --- Row 2: Performance metrics ---
    col1, col2, col3 = st.columns(3)
    with col1:
        reqs = metrics_data.get("requests_processed", 0) if metrics_available else 0
        st.metric("Requests Processed", f"{reqs:,}")
    with col2:
        avg_time = metrics_data.get("average_processing_time_ms", 0) if metrics_available else 0
        st.metric("Avg Response", f"{avg_time:.1f} ms")
    with col3:
        rps = metrics_data.get("requests_per_second", 0) if metrics_available else 0
        st.metric("Throughput", f"{rps:.1f} req/s")

    # --- Row 3: Prediction distribution ---
    st.subheader("Prediction Distribution")
    pred_history = report.get("prediction_distribution", [])
    if pred_history:
        import pandas as pd
        df_pred = pd.DataFrame(pred_history)
        if "mean" in df_pred.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(df_pred.set_index(df_pred.index)["mean"], use_container_width=True)
                st.caption("Mean fraud probability over time")
            with col2:
                st.bar_chart(
                    {"low": [df_pred["low_risk_ratio"].mean()],
                     "high": [df_pred["high_risk_ratio"].mean()]},
                    use_container_width=True,
                )
                st.caption("Avg low-risk vs high-risk ratio")
    else:
        st.info("Prediction distribution data will appear after predictions accumulate.")

    st.divider()

    # --- Row 4: Registered versions table ---
    st.subheader("Model Version History")
    all_versions = versions.get("versions", {})
    if all_versions:
        import pandas as pd
        rows = []
        for v, info in sorted(all_versions.items(), reverse=True):
            m = info.get("metrics", {})
            rows.append({
                "Version": v,
                "Samples": info.get("training_samples", "?"),
                "F1": m.get("f1_score", "?"),
                "AUC": m.get("roc_auc", "?"),
                "Created": info.get("created_at", "?")[:16],
                "Production": "Yes" if v == versions.get("production") else "",
                "Shadow": "Yes" if v == versions.get("shadow") else "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No model versions registered yet. Run train.py to create a version.")

    # --- Row 5: Alert summary ---
    st.subheader("Alerts & Recommendations")
    alerts = report.get("alerts", [])
    recs = report.get("recommendations", [])
    if alerts:
        for a in alerts:
            st.warning(a)
    if recs:
        for r in recs:
            st.info(r)
    if not alerts and not recs:
        st.success("No active alerts. System operating normally.")


def main():
    """Main dashboard function"""

    # Header
    st.title("[Fraud Detection] Fraud Detection Dashboard")
    st.markdown("Real-time monitoring for ML-powered fraud detection system")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # API Status check
    health_data, api_healthy = get_api_health()
    metrics_data, metrics_available = get_api_metrics()

    if api_healthy:
        st.sidebar.success("API Connected")
        st.sidebar.caption(f"Version: {health_data.get('version', 'unknown')}")
    else:
        st.sidebar.error("API Disconnected")

    # Page selector
    page = st.sidebar.radio(
        "Page",
        ["Transaction Monitor", "Model Health"],
    )

    if page == "Transaction Monitor":
        # Dashboard settings
        auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
        refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 30, 5)
        fraud_bias = st.sidebar.slider("Fraud simulation rate", 0.0, 0.5, 0.1, 0.01)

        # Initialize session state
        if "transaction_history" not in st.session_state:
            st.session_state.transaction_history = []
        if "simulator" not in st.session_state:
            st.session_state.simulator = TransactionSimulator()

        # Manual refresh button
        if st.sidebar.button("Refresh Now"):
            st.rerun()

    # Main dashboard content
    if not api_healthy:
        st.error(
            "Cannot connect to Fraud Detection API. Please ensure the API is running on http://localhost:8000"
        )
        st.info("To start the API, run: `uvicorn src.api.fraud_api:app --reload`")
        return

    if page == "Model Health":
        render_model_health(metrics_data, metrics_available)
        return

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if metrics_available:
            requests_processed = metrics_data.get("requests_processed", 0)
            st.metric(
                "Requests Processed",
                f"{requests_processed:,}",
                delta=f"+{random.randint(1, 5)}" if requests_processed > 0 else None,
            )
        else:
            st.metric("Requests Processed", "N/A")

    with col2:
        if metrics_available:
            avg_time = metrics_data.get("average_processing_time_ms", 0)
            st.metric(
                "Avg Response Time",
                f"{avg_time:.1f} ms",
                delta=f"{random.uniform(-2, 2):.1f} ms",
            )
        else:
            st.metric("Avg Response Time", "N/A")

    with col3:
        if metrics_available:
            rps = metrics_data.get("requests_per_second", 0)
            st.metric(
                "Requests/Second",
                f"{rps:.1f}",
                delta=f"{random.uniform(-0.5, 0.5):.1f}",
            )
        else:
            st.metric("Requests/Second", "N/A")

    with col4:
        uptime = metrics_data.get("uptime_seconds", 0) if metrics_available else 0
        uptime_hours = uptime / 3600
        st.metric(
            "System Uptime",
            f"{uptime_hours:.1f} hrs",
            delta="Online" if uptime > 0 else "Offline",
        )

    st.divider()

    # Real-time transaction monitoring
    st.header("📊 Real-Time Transaction Monitoring")

    # Generate transaction automatically on auto-refresh or manually via button
    should_generate = (
        st.button("Simulate New Transaction") or
        (auto_refresh and len(st.session_state.transaction_history) < 50)
    )

    if should_generate:
        # Generate transaction
        transaction = st.session_state.simulator.generate_transaction(fraud_bias)

        # Get prediction
        prediction, pred_success = predict_transaction(transaction)

        if pred_success:
            # Add to history
            transaction_record = {
                **transaction,
                **prediction,
                "timestamp": datetime.now(),
            }
            st.session_state.transaction_history.append(transaction_record)

            # Keep only last 100 transactions
            if len(st.session_state.transaction_history) > 100:
                st.session_state.transaction_history = (
                    st.session_state.transaction_history[-100:]
                )

            # Display latest transaction
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader(f"Latest Transaction: {transaction['transaction_id']}")

                details_col1, details_col2, details_col3 = st.columns(3)
                with details_col1:
                    st.metric("Amount", f"${transaction['Amount']:,.2f}")
                with details_col2:
                    fraud_prob = prediction.get("fraud_probability", 0)
                    st.metric("Fraud Probability", f"{fraud_prob:.1%}")
                with details_col3:
                    risk_level = prediction.get("risk_level", "UNKNOWN")
                    st.metric("Risk Level", risk_level)

                # Show key feature values that influence the prediction
                st.caption("Key features of this transaction:")
                feat_cols = st.columns(6)
                key_features = ["V1", "V4", "V10", "V12", "V14", "V17"]
                for i, feat in enumerate(key_features):
                    with feat_cols[i]:
                        val = transaction.get(feat, 0)
                        st.metric(feat, f"{val:.3f}")

            with col2:
                # Risk gauge
                fraud_prob = prediction.get("fraud_probability", 0)
                gauge_fig = create_gauge_chart(
                    fraud_prob, "Fraud Risk", max_value=1.0, color_thresholds=[0.3, 0.7]
                )
                st.plotly_chart(gauge_fig, use_container_width=True)

        else:
            st.error(
                f"Failed to get prediction: {prediction.get('error', 'Unknown error')}"
            )

    # Transaction history analysis
    if st.session_state.transaction_history:
        st.subheader("📈 Transaction History Analysis")

        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.transaction_history)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_transactions = len(df)
            st.metric("Total Transactions", total_transactions)

        with col2:
            fraud_detected = sum(df["is_fraud"])
            st.metric(
                "Fraud Detected",
                fraud_detected,
                delta=f"{fraud_detected/total_transactions:.1%}",
            )

        with col3:
            avg_amount = df["Amount"].mean()
            st.metric("Avg Amount", f"${avg_amount:,.2f}")

        with col4:
            high_risk = sum(df["risk_level"].isin(["HIGH", "CRITICAL"]))
            st.metric(
                "High Risk", high_risk, delta=f"{high_risk/total_transactions:.1%}"
            )

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Fraud probability distribution
            fig_dist = px.histogram(
                df,
                x="fraud_probability",
                nbins=20,
                title="Fraud Probability Distribution",
                labels={"fraud_probability": "Fraud Probability", "count": "Count"},
            )
            fig_dist.update_layout(height=400)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            # Risk level distribution
            risk_counts = df["risk_level"].value_counts()
            fig_risk = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Level Distribution",
                color_discrete_map={
                    "LOW": "#00CC00",
                    "MEDIUM": "#FFD700",
                    "HIGH": "#FF8C00",
                    "CRITICAL": "#FF4444",
                },
            )
            fig_risk.update_layout(height=400)
            st.plotly_chart(fig_risk, use_container_width=True)

        # Time series analysis
        if len(df) > 1:
            st.subheader("⏱️ Time Series Analysis")

            # Convert timestamp to datetime if it's not already
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            # Create time series plot
            fig_ts = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=[
                    "Transaction Amount Over Time",
                    "Fraud Probability Over Time",
                ],
                vertical_spacing=0.1,
            )

            # Amount over time
            fig_ts.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["Amount"],
                    mode="lines+markers",
                    name="Amount",
                    line=dict(color="blue"),
                ),
                row=1,
                col=1,
            )

            # Fraud probability over time
            colors = ["red" if is_fraud else "green" for is_fraud in df["is_fraud"]]
            fig_ts.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["fraud_probability"],
                    mode="markers",
                    name="Fraud Probability",
                    marker=dict(color=colors, size=8),
                ),
                row=2,
                col=1,
            )

            # Add fraud threshold line
            fig_ts.add_hline(y=0.5, line_dash="dash", line_color="red", row=2, col=1)

            fig_ts.update_layout(height=600, showlegend=True)
            fig_ts.update_xaxes(title_text="Time", row=2, col=1)
            fig_ts.update_yaxes(title_text="Amount ($)", row=1, col=1)
            fig_ts.update_yaxes(title_text="Fraud Probability", row=2, col=1)

            st.plotly_chart(fig_ts, use_container_width=True)

        # Recent transactions table
        st.subheader("📋 Recent Transactions")

        # Display last 10 transactions
        recent_df = df.tail(10)[
            [
                "transaction_id",
                "Amount",
                "fraud_probability",
                "risk_level",
                "is_fraud",
                "timestamp",
            ]
        ]
        recent_df = recent_df.sort_values("timestamp", ascending=False)

        # Format for display
        recent_df["Amount"] = recent_df["Amount"].apply(lambda x: f"${x:,.2f}")
        recent_df["fraud_probability"] = recent_df["fraud_probability"].apply(
            lambda x: f"{x:.1%}"
        )
        recent_df["timestamp"] = recent_df["timestamp"].dt.strftime("%H:%M:%S")

        # Color code based on risk
        def highlight_risk(row):
            if row["risk_level"] == "CRITICAL":
                return ["background-color: #ffcccc"] * len(row)
            elif row["risk_level"] == "HIGH":
                return ["background-color: #ffe6cc"] * len(row)
            elif row["risk_level"] == "MEDIUM":
                return ["background-color: #fff2cc"] * len(row)
            else:
                return ["background-color: #e6ffe6"] * len(row)

        styled_df = recent_df.style.apply(highlight_risk, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

    else:
        st.info(
            "No transaction data available. Click 'Simulate New Transaction' to start monitoring."
        )

    # Footer
    st.divider()
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**[Fraud Detection] Fraud Detection System**")
    with col2:
        st.markdown("**Built by Hughetta1**")
    with col3:
        st.markdown(
            f"**📅 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**"
        )


if __name__ == "__main__":
    main()
