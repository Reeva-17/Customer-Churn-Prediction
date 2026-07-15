"""
app.py

Streamlit deployment app for the Customer Churn Prediction project.

Loads the trained XGBoost model (models/xgboost_model.pkl) and the
StandardScaler (models/scaler.pkl), collects a single customer's details
through a form, runs the same preprocessing pipeline used during training
(preprocessing.py), and predicts churn probability using the F1-optimal
decision threshold of 0.60 found in notebooks/08_XGBoost.ipynb.

Also surfaces model comparison metrics (reports/model_comparison.csv) and
SHAP-based explainability, both the precomputed global plots
(images/shap_bar.png, images/shap_summary.png, images/shap_waterfall.png)
and a live per-prediction SHAP waterfall plot.
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap
import streamlit as st

from config import (
    CATEGORICAL_OPTIONS,
    NUMERIC_RANGES,
    DECISION_THRESHOLD,
    SHAP_BAR,
    SHAP_SUMMARY,
    SHAP_WATERFALL,
)
from utils import (
    load_model,
    load_scaler,
    load_model_comparison,
    predict_churn,
    explain_prediction,
    format_percent,
)

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css(path: str):
    try:
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


load_css("style.css")

st.markdown(
    """
    <div class="app-header">
        <h1>📉 Customer Churn Prediction</h1>
        <p>Predict the likelihood a telecom customer will churn, powered by an XGBoost model
        trained on the IBM Telco Customer Churn dataset, with SHAP-based explainability.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Load artifacts
# ----------------------------------------------------------------------

try:
    model = load_model()
    scaler = load_scaler()
except FileNotFoundError as e:
    st.error(
        "Could not load model artifacts. Make sure `models/xgboost_model.pkl` "
        "exists and run `python generate_scaler.py` to create `models/scaler.pkl`. "
        f"\n\nDetails: {e}"
    )
    st.stop()

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

with st.sidebar:
    st.header("About this app")
    st.write(
        "This app deploys the best-performing model (XGBoost, selected by "
        "F1-Score) from the Customer Churn Prediction project."
    )
    st.markdown(f"**Decision threshold:** {DECISION_THRESHOLD:.2f}")
    st.markdown(
        "A customer is classified as **likely to churn** when the predicted "
        "probability is at or above this threshold, which was chosen to "
        "maximize F1-Score on the held-out test set."
    )
    st.divider()
    st.caption("Built with Streamlit · XGBoost · SHAP")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------

predict_tab, insights_tab, about_tab = st.tabs(
    ["🔮 Predict", "📊 Model Insights", "ℹ️ About the Data"]
)

# ----------------------------------------------------------------------
# Predict tab
# ----------------------------------------------------------------------

with predict_tab:
    st.subheader("Enter Customer Details")

    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", CATEGORICAL_OPTIONS["gender"])
            senior_citizen_label = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", CATEGORICAL_OPTIONS["Partner"])
            dependents = st.selectbox("Has Dependents", CATEGORICAL_OPTIONS["Dependents"])

            st.markdown("**Account**")
            tenure = st.slider(
                "Tenure (months)",
                min_value=NUMERIC_RANGES["tenure"]["min"],
                max_value=NUMERIC_RANGES["tenure"]["max"],
                value=NUMERIC_RANGES["tenure"]["default"],
            )
            contract = st.selectbox("Contract", CATEGORICAL_OPTIONS["Contract"])
            paperless_billing = st.selectbox(
                "Paperless Billing", CATEGORICAL_OPTIONS["PaperlessBilling"]
            )
            payment_method = st.selectbox(
                "Payment Method", CATEGORICAL_OPTIONS["PaymentMethod"]
            )

        with col2:
            st.markdown("**Phone & Internet**")
            phone_service = st.selectbox(
                "Phone Service", CATEGORICAL_OPTIONS["PhoneService"]
            )
            multiple_lines = st.selectbox(
                "Multiple Lines", CATEGORICAL_OPTIONS["MultipleLines"]
            )
            internet_service = st.selectbox(
                "Internet Service", CATEGORICAL_OPTIONS["InternetService"]
            )
            online_security = st.selectbox(
                "Online Security", CATEGORICAL_OPTIONS["OnlineSecurity"]
            )
            online_backup = st.selectbox(
                "Online Backup", CATEGORICAL_OPTIONS["OnlineBackup"]
            )

        with col3:
            st.markdown("**Add-on Services**")
            device_protection = st.selectbox(
                "Device Protection", CATEGORICAL_OPTIONS["DeviceProtection"]
            )
            tech_support = st.selectbox(
                "Tech Support", CATEGORICAL_OPTIONS["TechSupport"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", CATEGORICAL_OPTIONS["StreamingTV"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", CATEGORICAL_OPTIONS["StreamingMovies"]
            )

            st.markdown("**Charges**")
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=NUMERIC_RANGES["MonthlyCharges"]["min"],
                max_value=NUMERIC_RANGES["MonthlyCharges"]["max"],
                value=NUMERIC_RANGES["MonthlyCharges"]["default"],
                step=0.5,
            )
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=NUMERIC_RANGES["TotalCharges"]["min"],
                max_value=NUMERIC_RANGES["TotalCharges"]["max"],
                value=NUMERIC_RANGES["TotalCharges"]["default"],
                step=10.0,
            )

        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    if submitted:
        raw_input = {
            "SeniorCitizen": 1 if senior_citizen_label == "Yes" else 0,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "gender": gender,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
        }

        try:
            result = predict_churn(model, scaler, raw_input)
        except ValueError as e:
            st.error(f"Invalid input: {e}")
            st.stop()

        probability = result["probability"]
        prediction = result["prediction"]
        risk_level = result["risk_level"]
        risk_class = risk_level.lower()

        st.divider()
        st.subheader("Prediction Result")

        result_col, gauge_col = st.columns([1, 1])

        with result_col:
            verdict = "⚠️ Likely to Churn" if prediction == 1 else "✅ Likely to Stay"
            st.markdown(
                f"""
                <div class="result-card risk-{risk_class}">
                    <h3>{verdict}</h3>
                    <p>Churn Probability: <strong>{format_percent(probability)}</strong></p>
                    <p>Risk Level:
                        <span class="risk-badge {risk_class}">{risk_level}</span>
                    </p>
                    <p style="font-size:0.85rem; color:#666;">
                        Decision threshold: {DECISION_THRESHOLD:.2f}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with gauge_col:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%"},
                    title={"text": "Churn Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#1a2b4c"},
                        "steps": [
                            {"range": [0, 30], "color": "#d4edda"},
                            {"range": [30, DECISION_THRESHOLD * 100], "color": "#fff3cd"},
                            {"range": [DECISION_THRESHOLD * 100, 100], "color": "#f8d7da"},
                        ],
                        "threshold": {
                            "line": {"color": "#c0392b", "width": 3},
                            "thickness": 0.85,
                            "value": DECISION_THRESHOLD * 100,
                        },
                    },
                )
            )
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Why this prediction? (SHAP)")
        with st.spinner("Computing SHAP explanation..."):
            try:
                shap_value = explain_prediction(model, result["input_row"])
                fig2, ax = plt.subplots(figsize=(9, 5))
                shap.plots.waterfall(shap_value, show=False)
                st.pyplot(fig2, clear_figure=True)
                st.caption(
                    "Features pushing the prediction toward churn (red) or away from "
                    "churn (blue), starting from the model's average predicted output."
                )
            except Exception as e:
                st.info(f"SHAP explanation unavailable for this prediction: {e}")

# ----------------------------------------------------------------------
# Model Insights tab
# ----------------------------------------------------------------------

with insights_tab:
    st.subheader("Model Comparison")
    st.write(
        "Five models were trained and evaluated; XGBoost was selected as the "
        "final model for its highest F1-Score while maintaining strong ROC-AUC."
    )

    try:
        comparison_df = load_model_comparison()
        st.dataframe(
            comparison_df.style.format(
                {
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1 Score": "{:.4f}",
                    "ROC-AUC": "{:.4f}",
                }
            ),
            use_container_width=True,
        )

        fig3 = go.Figure()
        fig3.add_trace(
            go.Bar(
                x=comparison_df["Model"],
                y=comparison_df["F1 Score"],
                marker_color="#1a2b4c",
                name="F1 Score",
            )
        )
        fig3.update_layout(
            title="F1 Score by Model",
            yaxis_title="F1 Score",
            height=400,
        )
        st.plotly_chart(fig3, use_container_width=True)
    except FileNotFoundError:
        st.warning("Model comparison report not found.")

    st.divider()
    st.subheader("Global Feature Importance (SHAP)")
    st.write(
        "These plots summarize how each feature affects predictions across "
        "the entire test set, computed with `shap.TreeExplainer` "
        "(see notebooks/10_SHAP.ipynb)."
    )

    shap_col1, shap_col2 = st.columns(2)
    with shap_col1:
        if SHAP_BAR.exists():
            st.image(str(SHAP_BAR), caption="Mean |SHAP value| by feature", use_container_width=True)
        if SHAP_WATERFALL.exists():
            st.image(
                str(SHAP_WATERFALL),
                caption="Example single-prediction waterfall (test set row 0)",
                use_container_width=True,
            )
    with shap_col2:
        if SHAP_SUMMARY.exists():
            st.image(
                str(SHAP_SUMMARY),
                caption="Feature impact distribution (beeswarm)",
                use_container_width=True,
            )

# ----------------------------------------------------------------------
# About tab
# ----------------------------------------------------------------------

with about_tab:
    st.subheader("About the Dataset & Project")
    st.markdown(
        """
This app is built on the **IBM Telco Customer Churn** dataset (7,043 customers,
21 features). The full workflow:

1. Data cleaning (missing `TotalCharges` rows dropped, `customerID` removed)
2. Feature encoding (one-hot encoding via `pd.get_dummies(drop_first=True)`)
3. Scaling numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) with `StandardScaler`
4. Training and comparing 5 models: Logistic Regression, Decision Tree, SVM, Random Forest, XGBoost
5. Selecting XGBoost as the final model based on F1-Score
6. Tuning the decision threshold to **0.60** to maximize F1-Score on the test set
7. Explaining predictions with SHAP (TreeExplainer)

Since churn ("Yes") is the minority class, F1-Score was used as the primary
selection metric instead of raw accuracy.
        """
    )

st.markdown(
    '<div class="footer-note">Customer Churn Prediction · XGBoost + SHAP · Streamlit</div>',
    unsafe_allow_html=True,
)