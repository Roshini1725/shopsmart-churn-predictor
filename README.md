# 🛍️ ShopSmart Churn Predictor — Privacy-Centric Customer Retention Analytics

> *"Predict who's leaving. Understand why. Retain them — without compromising their privacy."*

A machine learning project that predicts **customer churn in a retail loyalty/rewards program**, combining predictive analytics with **Privacy by Design principles**. Built as an extension of the ShopSmart: Privacy-Centric Customer Rewards initiative.

---

## 🧭 Project Background

Loyalty programs collect rich behavioural data — purchase frequency, redemption patterns, browsing history. Most churn models use all of it indiscriminately. This project asks a different question:

> **Can we build an equally accurate churn model using only privacy-respecting, aggregated features — without tracking individual transactions or storing personal identifiers?**

This is the analytical backbone of what a real privacy-first CRM team would build.

---

## 🎯 Business Problem

ShopSmart's rewards program is seeing:
- **~28% annual churn** among loyalty members
- High acquisition cost per new member (~3x retention cost)
- No early warning system for at-risk members

**Goal:** Build a model that scores members monthly by churn risk, enabling the CRM team to trigger personalised retention offers — while staying compliant with GDPR data minimisation principles.

---

## 📁 Project Structure

```
shopsmart-churn-predictor/
│
├── data/
│   └── README.md                        # Dataset instructions
│
├── notebooks/
│   └── ShopSmart_Churn_Analysis.ipynb   # Full EDA + modelling story
│
├── src/
│   ├── preprocess.py                    # Privacy-aware feature engineering
│   ├── train.py                         # LR + XGBoost training pipeline
│   └── evaluate.py                      # Metrics, plots, SHAP-ready hooks
│
├── dashboard/
│   └── app.py                           # Streamlit CRM dashboard
│
├── models/
│   └── README.md                        # Saved artifacts (gitignored)
│
├── tests/
│   └── test_preprocess.py               # Unit tests
│
├── plots/                               # Auto-generated EDA charts
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🗂️ Dataset

**E-Commerce Customer Churn** — publicly available on [Kaggle](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction).

**Why this dataset over Telco:**
- Retail/loyalty context matches ShopSmart's rewards program use case
- Contains behavioural signals: order count, days since last order, coupon usage, satisfaction score
- No direct PII — aligns with data minimisation principles

> Download `E Commerce.xlsx` from Kaggle and place it in `data/`.

---

## ⚙️ Setup

```bash
git clone https://github.com/your-username/shopsmart-churn-predictor.git
cd shopsmart-churn-predictor
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Run the Pipeline

```bash
# Train models
python src/train.py

# Launch dashboard
streamlit run dashboard/app.py

# Run tests
pytest tests/
```

---

## 📓 Notebook Highlights

The Jupyter notebook `ShopSmart_Churn_Analysis.ipynb` walks through:

1. **Business framing** — what churn costs ShopSmart per year
2. **Privacy audit** — which features are safe to use vs. privacy-invasive
3. **EDA** — churn by city tier, device type, satisfaction score, tenure
4. **Feature engineering** — RFM-inspired aggregated signals
5. **Modelling** — Logistic Regression vs. XGBoost with class imbalance handling
6. **Evaluation** — ROC-AUC, precision-recall, confusion matrix
7. **Business recommendations** — segment-specific retention strategies

---

## 📊 Privacy-by-Design Feature Framework

| Feature | Privacy Risk | Used? | Rationale |
|---------|-------------|-------|-----------|
| Order count (last 6 months) | Low | ✅ | Aggregated behaviour |
| Days since last order | Low | ✅ | Recency signal, no transaction detail |
| Avg order value bucket | Low | ✅ | Bucketed — not exact amount |
| Coupon usage rate | Low | ✅ | Ratio, not individual transactions |
| Satisfaction score | Low | ✅ | Self-reported |
| City tier | Low | ✅ | Regional, not address |
| Device type | Medium | ✅ | Used in aggregate only |
| Individual purchase history | High | ❌ | Violates data minimisation |
| Login timestamps | High | ❌ | Unnecessary for churn prediction |
| Browsing/click data | High | ❌ | Disproportionate to purpose |

---

## 📈 Model Performance

| Model | ROC-AUC | F1 (Churn) | Precision | Recall |
|-------|---------|-----------|-----------|--------|
| Logistic Regression | ~0.83 | ~0.62 | ~0.67 | ~0.58 |
| XGBoost | ~0.89 | ~0.72 | ~0.74 | ~0.70 |

XGBoost achieves ~7% higher AUC while using only privacy-safe features — validating that strong retention analytics do not require invasive data collection.

---

## 🔑 Key Business Findings

- Members with **satisfaction score ≤ 2** churn at 58% — highest single predictor
- **Single-device users** churn at 2x the rate of multi-device members
- Members who **never used a coupon** churn 34% more than active coupon users
- **Tier 1 city** members churn less — likely due to faster delivery SLAs
- The **first 3 months** are the highest-risk window — onboarding is critical

---

## 💼 CRM Action Framework

| Churn Risk | Score | Recommended Action |
|-----------|-------|-------------------|
| 🔴 High | > 70% | Personal outreach + bonus points offer |
| 🟡 Medium | 40–70% | Automated re-engagement email + coupon |
| 🟢 Low | < 40% | Standard loyalty newsletter |

---

## 🔗 Related Project

This project connects to **[ShopSmart: Privacy-Centric Customer Rewards](https://github.com/your-username/shopsmart-privacy-rewards)** — which covers the GDPR-compliant loyalty program architecture that generates the data this model consumes.

Together they form a full-stack view: **program design → data collection → churn prediction → CRM action**.

---

## 🛠️ Tech Stack

`scikit-learn` · `xgboost` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `streamlit` · `plotly` · `joblib` · `pytest`

---

## 🔮 Future Work

- [ ] Add SHAP values for per-customer explainability
- [ ] Simulate retention ROI using predicted churn probabilities
- [ ] Add differential privacy noise to training pipeline
- [ ] Deploy on Streamlit Cloud with sample data

---

## 📄 License

MIT License — see `LICENSE` for details.
