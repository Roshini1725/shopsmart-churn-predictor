"""
app.py — ShopSmart Churn Predictor Dashboard
--------------------------------------------
Run:  streamlit run dashboard/app.py
"""

import os, sys, joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import load_data, privacy_audit, clean_data, engineer_features, encode_features

BASE       = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, "..", "models")
DATA_PATH  = os.path.join(BASE, "..", "data", "E Commerce.xlsx")

st.set_page_config(page_title="ShopSmart Churn Predictor", page_icon="🛍️", layout="wide")

# ── Brand colours ──────────────────────────────────────────────────────────────
RED  = "#E63946"
BLUE = "#457B9D"
GREEN = "#2A9D8F"

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/shopping-bag.png", width=60)
st.sidebar.title("ShopSmart\nChurn Predictor")
st.sidebar.caption("Privacy-Centric Retention Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Segment Analysis",
    "🔍 Member Risk Score",
    "🏆 Model Performance",
    "🔒 Privacy Dashboard",
])

# ── Load resources ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    lr   = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    xgb  = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    return lr, xgb, cols

@st.cache_data
def load_processed():
    df_raw  = load_data(DATA_PATH)
    df_priv = privacy_audit(df_raw, verbose=False)
    df_clean = clean_data(df_priv)
    df_feat  = engineer_features(df_clean)
    df_enc   = encode_features(df_feat)
    return df_raw, df_enc

try:
    lr_model, xgb_model, feature_cols = load_models()
    df_raw, df_enc = load_processed()
    loaded = True
except FileNotFoundError:
    loaded = False
    st.warning("⚠️ Run `python src/train.py` first to generate model artifacts.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Segment Analysis
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Segment Analysis":
    st.title("📊 Member Churn by Segment")

    c1, c2, c3, c4 = st.columns(4)
    churn_rate = df_raw["Churn"].mean()
    churned    = df_raw["Churn"].sum()
    c1.metric("Total Members",    f"{len(df_raw):,}")
    c2.metric("Churned Members",  f"{int(churned):,}")
    c3.metric("Churn Rate",       f"{churn_rate:.1%}")
    c4.metric("Avg Satisfaction", f"{df_raw['SatisfactionScore'].mean():.1f} / 5")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Satisfaction Score")
        sat = df_raw.groupby("SatisfactionScore")["Churn"].mean().reset_index()
        sat.columns = ["Score", "Churn Rate"]
        fig = px.bar(sat, x="Score", y="Churn Rate", text_auto=".1%",
                     color="Churn Rate", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("City Tier")
        ct = df_raw.groupby("CityTier")["Churn"].mean().reset_index()
        ct["CityTier"] = ct["CityTier"].map({1:"Tier 1 (Metro)", 2:"Tier 2 (Mid)", 3:"Tier 3 (Small)"})
        ct.columns = ["City Tier", "Churn Rate"]
        fig = px.bar(ct, x="City Tier", y="Churn Rate", text_auto=".1%",
                     color="Churn Rate", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tenure Distribution by Churn")
    fig = px.histogram(df_raw, x="Tenure", color=df_raw["Churn"].map({1:"Churned", 0:"Retained"}),
                       barmode="overlay", nbins=30,
                       color_discrete_map={"Churned": RED, "Retained": BLUE},
                       labels={"color": "Status"})
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Member Risk Score
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Member Risk Score":
    st.title("🔍 Individual Member Risk Scorer")
    st.caption("Enter member behavioural data to get a privacy-safe churn probability.")

    if not loaded: st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        tenure         = st.slider("Tenure (months)", 0, 60, 6)
        satisfaction   = st.slider("Satisfaction Score (1-5)", 1, 5, 3)
        order_count    = st.slider("Orders (last 6 months)", 0, 20, 4)
        day_last_order = st.slider("Days Since Last Order", 0, 30, 3)

    with col2:
        city_tier    = st.selectbox("City Tier", [1, 2, 3], format_func=lambda x: f"Tier {x}")
        device       = st.selectbox("Preferred Login Device", ["Mobile Phone", "Computer"])
        complain     = st.selectbox("Raised Complaint?", ["No", "Yes"])
        coupon_used  = st.slider("Coupons Used", 0, 10, 1)

    with col3:
        order_cat   = st.selectbox("Preferred Category", ["Laptop & Accessory", "Mobile", "Fashion", "Grocery", "Others"])
        payment     = st.selectbox("Payment Mode", ["Debit Card", "UPI", "Credit Card", "Cash on Delivery", "E wallet"])
        num_devices = st.slider("Devices Registered", 1, 6, 2)
        num_address = st.slider("Saved Addresses", 1, 10, 2)

    if st.button("🔮 Score This Member", type="primary"):
        input_row = {
            "Tenure": [tenure], "CityTier": [city_tier],
            "WarehouseToHome": [15], "NumberOfDeviceRegistered": [num_devices],
            "SatisfactionScore": [satisfaction], "NumberOfAddress": [num_address],
            "Complain": [1 if complain == "Yes" else 0],
            "OrderAmountHikeFromlastYear": [15],
            "CouponUsed": [coupon_used], "OrderCount": [order_count],
            "DaySinceLastOrder": [day_last_order],
            "PreferredLoginDevice": [device],
            "PreferredPaymentMode": [payment],
            "Gender": ["Male"],
            "PreferedOrderCat": [order_cat],
            "MaritalStatus": ["Single"],
            "Churn": [0],
        }
        row_df   = pd.DataFrame(input_row)
        row_feat = engineer_features(row_df)
        row_enc  = encode_features(row_feat)
        row_enc.drop(columns=["Churn"], errors="ignore", inplace=True)
        for col in feature_cols:
            if col not in row_enc.columns:
                row_enc[col] = 0
        row_enc = row_enc[feature_cols]

        xgb_prob = xgb_model.predict_proba(row_enc)[0][1]
        lr_prob  = lr_model.predict_proba(row_enc)[0][1]

        tier = "🔴 HIGH RISK" if xgb_prob > 0.6 else ("🟡 MEDIUM RISK" if xgb_prob > 0.3 else "🟢 LOW RISK")
        action = (
            "Personal outreach + 500 bonus points offer" if xgb_prob > 0.6
            else "Automated re-engagement email + 10% coupon" if xgb_prob > 0.3
            else "Standard loyalty newsletter — no action needed"
        )

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("XGBoost Churn Probability", f"{xgb_prob:.1%}")
        m2.metric("Logistic Regression",        f"{lr_prob:.1%}")
        m3.metric("Risk Tier", tier)

        st.info(f"**Recommended CRM Action:** {action}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=xgb_prob * 100,
            title={"text": "Churn Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": RED if xgb_prob > 0.5 else BLUE},
                "steps": [
                    {"range": [0,  30],  "color": "#d1fae5"},
                    {"range": [30, 60],  "color": "#fef3c7"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
            },
        ))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Model Performance":
    st.title("🏆 Model Performance Comparison")
    if not loaded: st.stop()

    from sklearn.model_selection import train_test_split
    X = df_enc.drop(columns=["Churn"])
    y = df_enc["Churn"]
    for col in feature_cols:
        if col not in X.columns: X[col] = 0
    X = X[feature_cols]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    lr_proba  = lr_model.predict_proba(X_test)[:, 1]
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    lr_auc    = roc_auc_score(y_test, lr_proba)
    xgb_auc   = roc_auc_score(y_test, xgb_proba)

    c1, c2 = st.columns(2)
    c1.metric("Logistic Regression AUC", f"{lr_auc:.4f}")
    c2.metric("XGBoost AUC", f"{xgb_auc:.4f}", delta=f"+{xgb_auc-lr_auc:.4f} vs LR")

    fpr_lr,  tpr_lr,  _ = roc_curve(y_test, lr_proba)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr_lr,  y=tpr_lr,  name=f"LR (AUC={lr_auc:.3f})",  line=dict(color=BLUE)))
    fig.add_trace(go.Scatter(x=fpr_xgb, y=tpr_xgb, name=f"XGB (AUC={xgb_auc:.3f})", line=dict(color=RED)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random", line=dict(color="gray", dash="dash")))
    fig.update_layout(title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("XGBoost — Top 15 Feature Importance")
    fi = pd.Series(xgb_model.feature_importances_, index=feature_cols).nlargest(15).sort_values()
    fig2 = px.bar(fi, orientation="h", color=fi.values, color_continuous_scale="Blues",
                  labels={"value": "Importance", "index": "Feature"})
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Privacy Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔒 Privacy Dashboard":
    st.title("🔒 Privacy by Design — Feature Audit")
    st.markdown("Transparency report on how member data is used in this model.")

    audit_data = {
        "Feature": [
            "Tenure", "OrderCount", "DaySinceLastOrder", "SatisfactionScore",
            "CouponUsed", "CityTier", "NumberOfDeviceRegistered",
            "Individual Transactions", "Login Timestamps", "Browsing History",
        ],
        "Privacy Risk": ["Low","Low","Low","Low","Low","Low","Medium","High","High","High"],
        "Used in Model": ["✅","✅","✅","✅","✅","✅","✅","❌","❌","❌"],
        "Rationale": [
            "Aggregated — no PII",
            "Count only — no item detail",
            "Recency signal — no timestamp",
            "Self-reported",
            "Ratio — not transaction detail",
            "Regional — not address",
            "Count only — no device ID",
            "Violates data minimisation",
            "Not needed for churn prediction",
            "Disproportionate to purpose",
        ],
    }

    df_audit = pd.DataFrame(audit_data)
    st.dataframe(df_audit, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("GDPR Compliance Notes")
    st.markdown("""
    - **Art. 5(1)(c) — Data Minimisation:** Only features directly relevant to churn prediction are used.
    - **Art. 22 — Automated Decision-Making:** Members have the right to request human review of their risk score. Use the Logistic Regression model output for explainability in such cases.
    - **Art. 17 — Right to Erasure:** CustomerID is never stored in model artifacts. Removal from source data is sufficient.
    - **Art. 25 — Privacy by Design:** Feature selection is documented and reviewed before each model retraining cycle.
    """)
