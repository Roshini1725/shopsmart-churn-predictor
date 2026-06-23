"""
evaluate.py
-----------
Evaluation plots: ROC, confusion matrix, feature importance.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
BRAND_RED  = "#E63946"
BRAND_BLUE = "#457B9D"


def _ensure():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, model_name, save=True):
    _ensure()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"], ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(PLOTS_DIR, f"cm_{model_name.lower().replace(' ','_')}.png"), dpi=150)
    return fig


def plot_roc_curves(models_data, save=True):
    _ensure()
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = [BRAND_BLUE, BRAND_RED]
    for i, m in enumerate(models_data):
        fpr, tpr, _ = roc_curve(m["y_true"], m["y_proba"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % 2], lw=2,
                label=f"{m['name']} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontweight="bold"); ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"), dpi=150)
    return fig


def plot_feature_importance(model, feature_names, top_n=15, save=True):
    _ensure()
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in idx[::-1]],
            importances[idx[::-1]], color=BRAND_BLUE)
    ax.set_title(f"Top {top_n} Features — XGBoost", fontweight="bold")
    ax.set_xlabel("Feature Importance (Gain)")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
    return fig
