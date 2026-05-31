"""
Model Training: XGBoost + LightGBM with cross-validation, MLflow tracking, SHAP.
"""
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_models(config: dict) -> Dict:
    """Instantiate XGBoost and LightGBM models from config."""
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    xgb_params = config["training"]["models"]["xgboost"].copy()
    lgbm_params = config["training"]["models"]["lightgbm"].copy()

    models = {
        "XGBoost": XGBClassifier(**xgb_params, verbosity=0),
        "LightGBM": LGBMClassifier(**lgbm_params, verbose=-1),
    }
    return models


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute full classification metrics."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }
    return metrics, y_prob, y_pred


def plot_confusion_matrix(y_test, y_pred, model_name: str, save_dir: str):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_shap_summary(model, X_test: np.ndarray, feature_names: list, model_name: str, save_dir: str):
    """Generate SHAP summary plot for explainability."""
    try:
        logger.info(f"Computing SHAP values for {model_name}...")
        explainer = shap.TreeExplainer(model)
        # Use a sample for speed
        sample_size = min(500, len(X_test))
        idx = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test[idx]

        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 for binary

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names[:X_sample.shape[1]],
            show=False, max_display=15,
        )
        plt.title(f"{model_name} - SHAP Feature Importance")
        plt.tight_layout()
        path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_shap_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP plot saved: {path}")
        return path
    except Exception as e:
        logger.warning(f"SHAP plot failed for {model_name}: {e}")
        return None


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: dict,
    feature_names: list,
) -> Tuple[object, Dict, str]:
    """Train all models, track with MLflow, return best model."""
    import mlflow

    os.makedirs("models", exist_ok=True)
    os.makedirs("models/plots", exist_ok=True)

    mlflow_cfg = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    models = get_models(config)
    cv = StratifiedKFold(n_splits=config["training"]["cv_folds"], shuffle=True, random_state=42)
    threshold = config["training"]["threshold"]

    results = {}
    best_auc = 0.0
    best_model = None
    best_model_name = None

    for model_name, model in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name}...")

        with mlflow.start_run(run_name=model_name):
            # Log hyperparameters
            params_key = model_name.lower().replace(" ", "")
            params_key = "xgboost" if "xg" in params_key else "lightgbm"
            for k, v in config["training"]["models"][params_key].items():
                mlflow.log_param(k, v)

            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, scoring="roc_auc", n_jobs=-1,
            )
            logger.info(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

            # Final training
            model.fit(X_train, y_train)

            # Evaluation
            metrics, y_prob, y_pred = evaluate_model(model, X_test, y_test, threshold)
            results[model_name] = metrics

            logger.info(f"  Test ROC-AUC: {metrics['roc_auc']}")
            logger.info(f"  Accuracy:     {metrics['accuracy']}")
            logger.info(f"  Precision:    {metrics['precision']}")
            logger.info(f"  Recall:       {metrics['recall']}")
            logger.info(f"  F1-Score:     {metrics['f1']}")

            for metric_name, val in metrics.items():
                mlflow.log_metric(metric_name, val)

            # Plots
            cm_path = plot_confusion_matrix(y_test, y_pred, model_name, "models/plots")
            mlflow.log_artifact(cm_path)

            shap_path = plot_shap_summary(model, X_test, feature_names, model_name, "models/plots")
            if shap_path:
                mlflow.log_artifact(shap_path)

            # Save model
            model_path = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)

            # Track best
            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_model = model
                best_model_name = model_name

    # Save best model
    best_path = "models/best_model.pkl"
    joblib.dump(best_model, best_path)
    logger.info(f"\nBest model: {best_model_name} (ROC-AUC={best_auc:.4f})")
    logger.info(f"Saved best model to {best_path}")

    # Save metrics summary
    summary = {
        "best_model": best_model_name,
        "best_roc_auc": best_auc,
        "all_results": results,
    }
    with open("models/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return best_model, results, best_model_name


def run_training(config_path: str = "config.yaml"):
    """Full training pipeline entry point."""
    from src.features.engineering import prepare_data, get_feature_names

    config = load_config(config_path)

    logger.info("Loading processed data...")
    df = pd.read_csv(config["data"]["processed_path"])

    # Split first to avoid leakage
    target = config["features"]["target_column"]
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(
        df,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=df[target],
    )

    logger.info(f"Train size: {len(df_train)}, Test size: {len(df_test)}")

    X_train, y_train, preprocessor = prepare_data(df_train, config, fit=True)
    X_test, y_test, _ = prepare_data(df_test, config, preprocessor=preprocessor, fit=False)

    feature_names = get_feature_names(preprocessor, config)

    # Save preprocessor
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    logger.info("Saved preprocessor to models/preprocessor.pkl")

    best_model, results, best_name = train_and_evaluate(
        X_train, X_test, y_train, y_test, config, feature_names
    )

    logger.info("\n" + "="*50)
    logger.info("Training complete!")
    logger.info(f"Best model: {best_name}")
    for name, metrics in results.items():
        logger.info(f"  {name}: AUC={metrics['roc_auc']}, F1={metrics['f1']}")

    return best_model, results


if __name__ == "__main__":
    run_training()
