"""
train.py
--------
Train Logistic Regression and XGBoost churn classifiers for ShopSmart.
"""

import os
import sys
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import run_pipeline

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "E Commerce.xlsx")


def train_logistic_regression(X_train, y_train):
    print("\n── Logistic Regression ──────────────────────────")
    model = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    print("Done.")
    return model


def train_xgboost(X_train, y_train):
    print("\n── XGBoost ──────────────────────────────────────")
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=neg / pos, subsample=0.8,
        colsample_bytree=0.8, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    print("Done.")
    return model


def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n{name} — ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
    return {"name": name, "auc": auc, "proba": y_proba}


def save(obj, filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(obj, path)
    print(f"Saved → {path}")


def main():
    X_train, X_test, y_train, y_test = run_pipeline(DATA_PATH)

    lr = train_logistic_regression(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    evaluate(lr, X_test, y_test, "Logistic Regression")
    evaluate(xgb, X_test, y_test, "XGBoost")

    save(lr,  "logistic_regression.pkl")
    save(xgb, "xgboost_model.pkl")
    save(list(X_train.columns), "feature_columns.pkl")

    print("\n✅ All models saved.")


if __name__ == "__main__":
    main()
