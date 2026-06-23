# Dataset

## E-Commerce Customer Churn Dataset

Download from [Kaggle](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction).

Place `E Commerce.xlsx` in this `data/` folder before running the pipeline.

### Why this dataset?
- Retail/loyalty context matches ShopSmart's rewards program
- Contains behavioural signals (order count, days since last order, satisfaction score, coupon usage)
- No direct PII — aligns with GDPR data minimisation principles

### Key Columns Used

| Column | Description |
|--------|-------------|
| Tenure | Months as a loyalty member |
| SatisfactionScore | Self-reported (1–5) |
| OrderCount | Orders in last month |
| DaySinceLastOrder | Recency signal |
| CouponUsed | Engagement signal |
| CityTier | Regional tier (1=Metro, 3=Small) |
| Complain | Whether member raised a complaint |
| Churn | **Target** (1=churned, 0=retained) |

### Excluded for Privacy
- Individual transaction records
- Login timestamps
- Browsing/click history
- CustomerID (dropped at pipeline entry)
