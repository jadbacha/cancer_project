# workshop-2: Cancer Prediction (Classification & Regression)

A Breast-Cancer Prediction Project

This project builds a complete, reproducible pipeline on the Breast Cancer Wisconsin dataset to:
- Classify tumors as malignant or benign.
- Regress continuous tumor-related features (e.g., `radius_mean`, `area_mean`).
- Produce clear figures and metrics ready for presentation.

## Project Structure
- `src/cancer_pipeline.py` – Loads data, runs EDA, trains models, saves metrics and figures.
- `reports/` – Generated artifacts:
  - `metrics.json` – Classification and regression metrics.
  - `feature_importances_rf.csv` – Top features from RandomForest.
  - `figures/` – Plots: target distribution, correlation heatmap, pairplot, confusion matrices, ROC curves, residuals.
- `requirements.txt` – Dependencies.
- `Breast_Cancer_Visualization_Preprocessing.ipynb` – Existing notebook (optional).

## How to Run
1. Create a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train and generate artifacts (EDA, metrics, figures):
   ```bash
   python src/cancer_pipeline.py --task both
   ```
   - Classification only:
     ```bash
     python src/cancer_pipeline.py --task classification
     ```
   - Regression only (choose target feature):
     ```bash
     python src/cancer_pipeline.py --task regression --reg-target radius_mean
     ```

4. Predict from a CSV (columns must match features):
   - Classification:
     ```bash
     python src/cancer_pipeline.py --task classification --predict-file path/to/unlabeled.csv
     ```
   - Regression (specify target to which the regressor was trained):
     ```bash
     python src/cancer_pipeline.py --task regression --reg-target area_mean --predict-file path/to/unlabeled.csv
     ```
   - Run both predictions:
     ```bash
     python src/cancer_pipeline.py --task both --reg-target radius_mean --predict-file path/to/unlabeled.csv
     ```

Artifacts will be written to `reports/`, `reports/figures/`, and trained models to `artifacts/`.

## What the Pipeline Does
- Loads the dataset from `sklearn.datasets.load_breast_cancer` and standardizes column names to underscore style (e.g., `radius_mean`).
- EDA:
  - Target class distribution.
  - Correlation heatmap of mean features.
  - Pairplot over selected features.
- Classification:
  - Models: Logistic Regression, SVM (RBF), RandomForest.
  - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC; confusion matrices; ROC curves.
  - Cross-validated ROC-AUC summary.
  - Top 20 feature importances from RandomForest.
- Regression:
  - Targets: `radius_mean` and `area_mean` predicted from the remaining features.
  - Models: Standardized Linear Regression, RandomForestRegressor.
  - Metrics: MAE, RMSE, R²; residuals plot for best model.

## Presenting the Results
- Use `reports/metrics.json` for headline numbers.
- Show figures from `reports/figures/`:
  - `target_distribution.png` – Class balance.
  - `correlation_heatmap_mean.png` – Feature relationships.
  - `pairplot_selected.png` – Feature separation by class.
  - `confusion_<model>.png` – Model error patterns.
  - `roc_<model>.png` – Diagnostic performance.
  - `feature_importances_rf.png` – Key predictors.
  - `residuals_<best>_<target>.png` – Regression fit quality.

## Notes
- The dataset ships with scikit-learn, so no external files are needed.
- If you prefer to present in a Jupyter Notebook, install the requirements and open `Breast_Cancer_Visualization_Preprocessing.ipynb` or create a new notebook that imports and runs `src/cancer_pipeline.py`.
