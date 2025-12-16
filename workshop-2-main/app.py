"""Streamlit UI for Cancer Prediction

This app provides an interactive front-end for the classification and
regression workflows defined in `src/cancer_pipeline.py`.

Features
--------
- Load sample or uploaded CSVs and run predictions.
- Manual single-row input (table or form) for quick checks.
- One-click training to rebuild models and refresh artifacts.
- Display headline metrics and key figures from `reports/`.
"""

import os
import io
import json
import datetime as dt
from typing import List, Dict, Optional

import streamlit as st
import pandas as pd
from joblib import load

# Project paths
ARTIFACTS_DIR = "artifacts"
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
SAMPLE_CSV = os.path.join("data", "sample_unlabeled.csv")

# Expected feature columns, aligned with normalized names in the pipeline
EXPECTED_FEATURES: List[str] = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se",
    "concave_points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst",
    "area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst",
    "fractal_dimension_worst"
]

# For regression predictions, exclude the target column from the input features
def regression_feature_list(target: str) -> List[str]:
    return [c for c in EXPECTED_FEATURES if c != target]

# App layout and header
st.set_page_config(page_title="Cancer Prediction Studio", page_icon="🩺", layout="wide")
st.title("🩺 Cancer Prediction Studio")
st.caption("A cute, high‑grade UI for classification and regression predictions")

# Sidebar controls for task selection and data source
st.sidebar.header("Controls")
task = st.sidebar.radio("Task", ["Classification", "Regression"], index=0)
reg_target = st.sidebar.selectbox("Regression target", ["radius_mean"], index=0,
                                  help="Pretrained target available: radius_mean")
use_sample = st.sidebar.checkbox("Use sample CSV (auto)", value=True)

# Utilities

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: lowercase and underscores, matching pipeline."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def ensure_expected_features(df: pd.DataFrame) -> pd.DataFrame:
    """Warn if required columns are missing and align dataframe columns."""
    df = normalize_columns(df)
    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing:
        st.warning(f"Missing expected columns: {missing[:6]}{'...' if len(missing)>6 else ''}")
        # Only keep existing expected columns; prediction will fail if required are missing
    return df[EXPECTED_FEATURES] if all(c in df.columns for c in EXPECTED_FEATURES) else df


def load_csv_from_upload(uploaded_file) -> Optional[pd.DataFrame]:
    """Read a CSV from Streamlit uploader, with error handling."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None


def read_metrics() -> Dict:
    """Load metrics summary from `reports/metrics.json` if present."""
    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def predict_classification(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Run classification predictions using the saved best classifier artifact."""
    model_path = os.path.join(ARTIFACTS_DIR, "classifier_best.joblib")
    if not os.path.exists(model_path):
        st.error("Classifier artifact not found. Run training first.")
        return None
    try:
        model = load(model_path)
        preds = model.predict(df[EXPECTED_FEATURES])
        out = df.copy()
        out["prediction"] = preds
        return out
    except Exception as e:
        st.error(f"Classification prediction failed: {e}")
        return None


def predict_regression(df: pd.DataFrame, target: str) -> Optional[pd.DataFrame]:
    """Run regression predictions using the saved regressor for `target`."""
    model_filename = f"regressor_{target}_best.joblib"
    model_path = os.path.join(ARTIFACTS_DIR, model_filename)
    if not os.path.exists(model_path):
        st.error(f"Regressor artifact '{model_filename}' not found. Run training first or switch target.")
        return None
    try:
        model = load(model_path)
        feature_cols = regression_feature_list(target)
        preds = model.predict(df[feature_cols])
        out = df.copy()
        out["prediction"] = preds
        return out
    except Exception as e:
        st.error(f"Regression prediction failed: {e}")
        return None


# Helper: build a one-row template for manual input
def feature_template() -> pd.DataFrame:
    """Create a single-row feature template prefilled from a dataset snapshot."""
    p = os.path.join(REPORTS_DIR, "breast_cancer_dataset_snapshot.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        df = normalize_columns(df)
        row = {c: float(df.iloc[0][c]) if c in df.columns else 0.0 for c in EXPECTED_FEATURES}
    else:
        row = {c: 0.0 for c in EXPECTED_FEATURES}
    return pd.DataFrame([row])


# Data input: sample CSV or upload
st.subheader("Data Input")
if use_sample and os.path.exists(SAMPLE_CSV):
    df_input = pd.read_csv(SAMPLE_CSV)
    st.success("Loaded sample data from 'data/sample_unlabeled.csv'")
else:
    uploaded = st.file_uploader("Upload CSV with feature columns", type=["csv"]) 
    df_input = load_csv_from_upload(uploaded) if uploaded is not None else None

if df_input is not None:
    df_input = ensure_expected_features(df_input)
    st.dataframe(df_input.head(), use_container_width=True)
else:
    st.info("Awaiting data upload or enable sample CSV.")

# Manual Input (single row) via table or form
st.subheader("Manual Input (one row)")
manual_mode = st.radio("Editor mode", ["Table", "Form"], index=0)

if manual_mode == "Table":
    manual_df = st.data_editor(feature_template(), use_container_width=True)
    mcol1, mcol2 = st.columns([1,1])
    with mcol1:
        if st.button("Predict (Manual) - Classification"):
            df_out = predict_classification(manual_df)
            if df_out is not None:
                st.success("Manual classification prediction complete!")
                st.dataframe(df_out, use_container_width=True)
    with mcol2:
        if st.button(f"Predict (Manual) - Regression [{reg_target}]"):
            df_out = predict_regression(manual_df, reg_target)
            if df_out is not None:
                st.success("Manual regression prediction complete!")
                st.dataframe(df_out, use_container_width=True)
else:
    defaults = feature_template().iloc[0].to_dict()
    cols = st.columns(3)
    values = {}
    for i, feat in enumerate(EXPECTED_FEATURES):
        with cols[i % 3]:
            values[feat] = st.number_input(feat, value=float(defaults.get(feat, 0.0)))
    manual_df = pd.DataFrame([values])
    mcol1, mcol2 = st.columns([1,1])
    with mcol1:
        if st.button("Predict (Manual Form) - Classification"):
            df_out = predict_classification(manual_df)
            if df_out is not None:
                st.success("Manual classification prediction complete!")
                st.dataframe(df_out, use_container_width=True)
    with mcol2:
        if st.button(f"Predict (Manual Form) - Regression [{reg_target}]"):
            df_out = predict_regression(manual_df, reg_target)
            if df_out is not None:
                st.success("Manual regression prediction complete!")
                st.dataframe(df_out, use_container_width=True)

# Actions: training and batch prediction
colA, colB, colC = st.columns([1,1,1])
with colA:
    train_cmd = "python src/cancer_pipeline.py --task both"
    if st.button("Rebuild Models (Train)"):
        with st.spinner("Training models... this takes a moment"):
            code = os.system(train_cmd)
        if code == 0:
            st.success("Training completed. Artifacts and metrics updated.")
        else:
            st.error("Training failed. Check terminal logs.")

with colB:
    if st.button("Predict"):
        if df_input is None:
            st.error("Please provide input data first.")
        else:
            with st.spinner("Running predictions..."):
                if task == "Classification":
                    df_out = predict_classification(df_input)
                else:
                    df_out = predict_regression(df_input, reg_target)
            if df_out is not None:
                st.success("Predictions complete!")
                st.dataframe(df_out.head(20), use_container_width=True)
                # Download
                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                fname = f"predictions_{task.lower()}_{ts}.csv"
                st.download_button("Download predictions CSV", data=csv_bytes, file_name=fname, mime="text/csv")

with colC:
    st.write("")

# Metrics & Visuals from reports
st.subheader("Model Report")
metrics = read_metrics()
if metrics:
    st.markdown("### Classification Metrics")
    if "classification" in metrics:
        cm = metrics["classification"]
        cols = st.columns(3)
        with cols[0]:
            st.metric(label="LogReg ROC-AUC", value=f"{cm['cv_roc_auc'].get('log_reg', '—'):.3f}")
            st.metric(label="Accuracy", value=f"{cm.get('log_reg',{}).get('accuracy', '—')}")
        with cols[1]:
            st.metric(label="SVM ROC-AUC", value=f"{cm['cv_roc_auc'].get('svm_rbf', '—'):.3f}")
            st.metric(label="Precision", value=f"{cm.get('log_reg',{}).get('precision', '—')}")
        with cols[2]:
            st.metric(label="RF ROC-AUC", value=f"{cm['cv_roc_auc'].get('rf', '—'):.3f}")
            st.metric(label="F1", value=f"{cm.get('log_reg',{}).get('f1', '—')}")

    st.markdown("### Regression Metrics (radius_mean)")
    if "regression_radius_mean" in metrics:
        rm = metrics["regression_radius_mean"]
        cols = st.columns(3)
        with cols[0]:
            st.metric("LinReg R²", f"{rm['lin_reg'].get('R2', '—'):.6f}")
        with cols[1]:
            st.metric("LinReg RMSE", f"{rm['lin_reg'].get('RMSE', '—'):.3f}")
        with cols[2]:
            st.metric("RF R²", f"{rm['rf_reg'].get('R2', '—'):.6f}")
else:
    st.info("Run training to populate metrics.")

st.markdown("### Visuals")
visual_paths = [
    os.path.join(FIGURES_DIR, "confusion_log_reg.png"),
    os.path.join(FIGURES_DIR, "roc_log_reg.png"),
    os.path.join(FIGURES_DIR, "residuals_lin_reg_radius_mean.png"),
    os.path.join(FIGURES_DIR, "feature_importances_rf.png"),
]

vcols = st.columns(2)
for i, p in enumerate(visual_paths):
    if os.path.exists(p):
        with vcols[i % 2]:
            # Display generated figures from the pipeline
            st.image(p, use_column_width=True)
    else:
        st.warning(f"Missing figure: {os.path.basename(p)}")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit, scikit‑learn, and the UCI Breast Cancer dataset."),
