"""Cancer Prediction Pipeline

This module provides a reproducible pipeline over the Breast Cancer
Wisconsin dataset using scikit-learn. It supports:

- Exploratory Data Analysis (EDA) to generate figures under `reports/figures/`.
- Classification (malignant vs benign) using Logistic Regression, SVM (RBF),
  and RandomForest, with metrics and cross-validated ROC-AUC.
- Regression for continuous targets (e.g., `radius_mean`, `area_mean`).
- CLI entry point to train models, generate artifacts/metrics, and optionally
  run predictions from a CSV file.

Artifacts (models) are saved under `artifacts/`, and metrics/figures under
`reports/`.
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC


def ensure_dirs():
    """Ensure output directories for reports and figures exist."""
    Path("reports").mkdir(exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)


def load_dataset():
    """Load breast cancer dataset and normalize column names.

    Returns
    -------
    tuple[pd.DataFrame, sklearn.utils.Bunch]
        The feature DataFrame with a `target` column and the original
        scikit-learn dataset object.
    """
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    # Rename columns to underscore style to match common Kaggle naming
    renamed = {
        "mean radius": "radius_mean",
        "mean texture": "texture_mean",
        "mean perimeter": "perimeter_mean",
        "mean area": "area_mean",
        "mean smoothness": "smoothness_mean",
        "mean compactness": "compactness_mean",
        "mean concavity": "concavity_mean",
        "mean concave points": "concave_points_mean",
        "mean symmetry": "symmetry_mean",
        "mean fractal dimension": "fractal_dimension_mean",
        "radius error": "radius_se",
        "texture error": "texture_se",
        "perimeter error": "perimeter_se",
        "area error": "area_se",
        "smoothness error": "smoothness_se",
        "compactness error": "compactness_se",
        "concavity error": "concavity_se",
        "concave points error": "concave_points_se",
        "symmetry error": "symmetry_se",
        "fractal dimension error": "fractal_dimension_se",
        "worst radius": "radius_worst",
        "worst texture": "texture_worst",
        "worst perimeter": "perimeter_worst",
        "worst area": "area_worst",
        "worst smoothness": "smoothness_worst",
        "worst compactness": "compactness_worst",
        "worst concavity": "concavity_worst",
        "worst concave points": "concave_points_worst",
        "worst symmetry": "symmetry_worst",
        "worst fractal dimension": "fractal_dimension_worst",
    }
    df = df.rename(columns=renamed)
    return df, data


def basic_eda(df: pd.DataFrame):
    """Generate basic EDA figures for the dataset.

    Produces target distribution, correlation heatmap (mean features), and
    a pairplot over selected features. Figures are saved under
    `reports/figures/`.
    """
    # Target distribution
    target_counts = df["target"].value_counts()
    plt.figure(figsize=(4, 3))
    sns.barplot(x=target_counts.index, y=target_counts.values)
    plt.title("Target distribution (0=Malignant, 1=Benign)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("reports/figures/target_distribution.png")
    plt.close()

    # Correlation heatmap (mean features only to keep it legible)
    mean_cols = [c for c in df.columns if c.endswith("_mean")] + ["target"]
    corr = df[mean_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation heatmap (mean features)")
    plt.tight_layout()
    plt.savefig("reports/figures/correlation_heatmap_mean.png")
    plt.close()

    # Pairplot of a few features
    few = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "target"]
    sns.pairplot(df[few], hue="target", diag_kind="kde")
    plt.savefig("reports/figures/pairplot_selected.png")
    plt.close()


def run_classification(df: pd.DataFrame):
    """Train classification models and produce metrics/figures.

    Trains Logistic Regression & SVM (with scaling) and RandomForest.
    Saves confusion matrices and ROC curves and computes per-model
    metrics including ROC-AUC. Also computes cross-validated ROC-AUC.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        A metrics dictionary and a DataFrame with RandomForest feature
        importances.
    """
    X = df.drop(columns=["target"])  # use all features
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "log_reg": Pipeline(
            steps=[("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
        ),
        "svm_rbf": Pipeline(
            steps=[("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True))]
        ),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42),
    }

    metrics = {}
    roc_curves = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            # fall back to decision function if available
            try:
                y_score = model.decision_function(X_test)
                # Scale raw decision function to 0..1 range for AUC comparability
                y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())
            except Exception:
                y_score = None

        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_score)) if y_score is not None else None,
        }

        # Confusion matrix figure
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"reports/figures/confusion_{name}.png")
        plt.close()

        # ROC curve
        if y_score is not None:
            # compute ROC points via sklearn
            from sklearn.metrics import RocCurveDisplay

            plt.figure(figsize=(4, 3))
            RocCurveDisplay.from_predictions(y_test, y_score)
            plt.title(f"ROC Curve: {name}")
            plt.tight_layout()
            plt.savefig(f"reports/figures/roc_{name}.png")
            plt.close()

    # Cross-validated ROC-AUC for pipelines with scaler
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {}
    for name, model in models.items():
        try:
            score = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()
            cv_scores[name] = float(score)
        except Exception:
            cv_scores[name] = None

    metrics["cv_roc_auc"] = cv_scores

    # Feature importance for RandomForest (trained on full data)
    rf_model = models["rf"]
    rf_model.fit(X, y)
    importances = rf_model.feature_importances_
    imp_df = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values(
        by="importance", ascending=False
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=imp_df.head(20), orient="h")
    plt.title("Top 20 Feature Importances (RandomForest)")
    plt.tight_layout()
    plt.savefig("reports/figures/feature_importances_rf.png")
    plt.close()

    return metrics, imp_df


def run_regression(df: pd.DataFrame, target: str = "radius_mean"):
    """Train regression models for a given continuous target.

    Fits a standardized Linear Regression and a RandomForestRegressor
    to predict the specified target from the remaining features.
    Saves a residuals plot for the best model by R².

    Returns
    -------
    dict
        Mapping of model name to MAE, RMSE, and R² scores.
    """
    # Predict a continuous feature (e.g., radius_mean) from all other features
    feature_cols = [c for c in df.columns if c != "target" and c != target]
    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "lin_reg": Pipeline(steps=[("scaler", StandardScaler()), ("reg", LinearRegression())]),
        "rf_reg": RandomForestRegressor(n_estimators=300, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "R2": float(r2_score(y_test, y_pred)),
        }

    # Residuals plot for best model by R²
    best = max(results.items(), key=lambda kv: kv[1]["R2"])[0]
    best_model = models[best]
    y_pred_best = best_model.predict(X_test)
    residuals = y_test - y_pred_best

    plt.figure(figsize=(5, 4))
    sns.scatterplot(x=y_pred_best, y=residuals)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"Residuals Plot ({best}) for target: {target}")
    plt.tight_layout()
    plt.savefig(f"reports/figures/residuals_{best}_{target}.png")
    plt.close()

    return results


def main():
    """Run the full pipeline: EDA, classification, regression, and summary save."""
    ensure_dirs()
    df, data = load_dataset()

    # Save raw data snapshot
    df.to_csv("reports/breast_cancer_dataset_snapshot.csv", index=False)

    # EDA Figures
    basic_eda(df)

    # Classification
    cls_metrics, imp_df = run_classification(df)

    # Regression (you can change target as needed)
    reg_metrics_radius = run_regression(df, target="radius_mean")
    reg_metrics_area = run_regression(df, target="area_mean")

    summary = {
        "classification": cls_metrics,
        "regression_radius_mean": reg_metrics_radius,
        "regression_area_mean": reg_metrics_area,
    }

    with open("reports/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Also export top features
    imp_df.to_csv("reports/feature_importances_rf.csv", index=False)

    print("\n=== Classification metrics ===")
    print(json.dumps(cls_metrics, indent=2))
    print("\n=== Regression metrics (radius_mean) ===")
    print(json.dumps(reg_metrics_radius, indent=2))
    print("\n=== Regression metrics (area_mean) ===")
    print(json.dumps(reg_metrics_area, indent=2))
    print("\nArtifacts saved in ./reports and ./reports/figures")


import argparse
import joblib
from datetime import datetime


def ensure_artifacts_dir():
    """Ensure model artifacts directory exists."""
    Path("artifacts").mkdir(exist_ok=True)


def train_best_classification_model(df: pd.DataFrame):
    """Train candidate classifiers and persist the best by ROC-AUC.

    Falls back to F1 if ROC-AUC cannot be computed (e.g., no scores).
    Saves the chosen estimator under `artifacts/classifier_best.joblib`.

    Returns
    -------
    tuple[str, float, object]
        Best model name, its score, and the fitted estimator.
    """
    X = df.drop(columns=["target"])  # use all features
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    candidates = {
        "log_reg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
        "svm_rbf": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True))]),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42),
    }
    best_name, best_score, best_model = None, -1.0, None
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        score = None
        try:
            # prefer ROC-AUC if available
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                y_score = model.decision_function(X_test)
            score = roc_auc_score(y_test, y_score)
        except Exception:
            score = f1_score(y_test, model.predict(X_test))
        if score > best_score:
            best_name, best_score, best_model = name, score, model
    joblib.dump(best_model, "artifacts/classifier_best.joblib")
    return best_name, best_score, best_model


def train_best_regression_model(df: pd.DataFrame, target: str):
    """Train candidate regressors for `target` and save the best by R².

    Returns
    -------
    tuple[str, float, object, str]
        Best model name, its R² score, the fitted estimator, and
        the artifact path where it was saved.
    """
    features = [c for c in df.columns if c != "target" and c != target]
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    candidates = {
        "lin_reg": Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]),
        "rf_reg": RandomForestRegressor(n_estimators=300, random_state=42),
    }
    best_name, best_score, best_model = None, -1.0, None
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        score = r2_score(y_test, model.predict(X_test))
        if score > best_score:
            best_name, best_score, best_model = name, score, model
    out_path = f"artifacts/regressor_{target}_best.joblib"
    joblib.dump(best_model, out_path)
    return best_name, best_score, best_model, out_path


def predict_from_csv(model_path: str, csv_path: str, task: str, target: str | None, df_template: pd.DataFrame):
    """Run predictions using a saved model on a CSV file.

    Validates that required feature columns are present, runs inference,
    and writes predictions CSV under `reports/`.
    """
    df_new = pd.read_csv(csv_path)
    # Build feature list based on template
    if task == "classification":
        feature_cols = [c for c in df_template.columns if c != "target"]
    else:
        feature_cols = [c for c in df_template.columns if c != "target" and c != target]
    missing = [c for c in feature_cols if c not in df_new.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    X_new = df_new[feature_cols]
    model = joblib.load(model_path)
    preds = model.predict(X_new)
    out = df_new.copy()
    out["prediction"] = preds
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_file = f"reports/predictions_{task}_{stamp}.csv"
    Path("reports").mkdir(exist_ok=True)
    out.to_csv(out_file, index=False)
    print(f"Saved predictions to: {out_file}")
    return out_file


def cli_main():
    """CLI entry point to train models, generate reports, and run predictions.

    Options
    -------
    --task: which task to execute (classification, regression, both)
    --reg-target: regression target feature name
    --predict-file: optional CSV to run inference against
    """
    parser = argparse.ArgumentParser(description="Cancer Prediction: classification and regression")
    parser.add_argument("--task", choices=["classification", "regression", "both"], default="both",
                        help="Which task to run")
    parser.add_argument("--reg-target", default="radius_mean",
                        help="Regression target feature (e.g., radius_mean, area_mean)")
    parser.add_argument("--predict-file", default=None,
                        help="Optional CSV file to run predictions on (columns must match features)")
    args = parser.parse_args()

    ensure_dirs()
    ensure_artifacts_dir()
    df, _ = load_dataset()

    # Always produce EDA when training (skip when only predicting)
    if args.predict_file is None or args.task == "both":
        basic_eda(df)

    if args.task in ("classification", "both"):
        print("Training classification models and selecting best...")
        best_name, best_score, best_model = train_best_classification_model(df)
        print(f"Best classifier: {best_name} (score={best_score:.4f}) → saved at artifacts/classifier_best.joblib")

    if args.task in ("regression", "both"):
        target = args.reg_target
        print(f"Training regression models for target '{target}' and selecting best...")
        best_name, best_score, best_model, out_path = train_best_regression_model(df, target)
        print(f"Best regressor: {best_name} (R2={best_score:.4f}) → saved at {out_path}")

    # Run the original summary pipeline when task==both and not only predicting
    if args.predict_file is None:
        main()

    # Inference on CSV if provided
    if args.predict_file:
        if args.task == "classification":
            predict_from_csv("artifacts/classifier_best.joblib", args.predict_file, "classification", None, df)
        elif args.task == "regression":
            predict_from_csv(f"artifacts/regressor_{args.reg_target}_best.joblib", args.predict_file, "regression", args.reg_target, df)
        else:
            # run both predictions, requires two outputs
            predict_from_csv("artifacts/classifier_best.joblib", args.predict_file, "classification", None, df)
            predict_from_csv(f"artifacts/regressor_{args.reg_target}_best.joblib", args.predict_file, "regression", args.reg_target, df)


if __name__ == "__main__":
    cli_main()
