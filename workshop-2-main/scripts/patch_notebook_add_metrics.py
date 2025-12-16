import nbformat as nbf
from pathlib import Path

nb_path = Path("/Users/charbel/cancer-project/Breast_Cancer_Visualization_Preprocessing.ipynb")

nb = nbf.read(nb_path, as_version=4)

# Sentinel to avoid duplicate insertion
sentinel_text = "Model Accuracy & Metrics"

already_present = any(
    (cell.get("source") or "").strip().startswith("# "+sentinel_text)
    for cell in nb.cells
)

if not already_present:
    md = nbf.v4.new_markdown_cell(
        "# " + sentinel_text + "\n\n"
        "These cells compute classification accuracy (and related metrics) "
        "from validation samples, and regression quality metrics (R², MAE, RMSE).\n\n"
        "- Classification uses `reports/validation_classification_samples.csv` (contains true vs predicted).\n"
        "- Regression uses `reports/validation_regression_samples.csv`.\n"
        "If you run the validation script again, these files will be refreshed automatically."
    )

    code_cls = nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n\n"
        "df_cls = pd.read_csv('reports/validation_classification_samples.csv')\n"
        "y_true = df_cls['true'].values\n"
        "y_pred = df_cls['pred'].values\n\n"
        "acc = accuracy_score(y_true, y_pred)\n"
        "prec = precision_score(y_true, y_pred)\n"
        "rec = recall_score(y_true, y_pred)\n"
        "f1 = f1_score(y_true, y_pred)\n\n"
        "print(f'Classification Accuracy: {acc:.4f}')\n"
        "print(f'Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}')\n"
    )

    code_reg = nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
        "import numpy as np\n\n"
        "df_reg = pd.read_csv('reports/validation_regression_samples.csv')\n"
        "y_true = df_reg['true'].values\n"
        "y_pred = df_reg['pred'].values\n\n"
        "mae = mean_absolute_error(y_true, y_pred)\n"
        "rmse = np.sqrt(mean_squared_error(y_true, y_pred))\n"
        "r2 = r2_score(y_true, y_pred)\n\n"
        "print(f'Regression R²: {r2:.6f}')\n"
        "print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}')\n"
    )

    nb.cells.extend([md, code_cls, code_reg])

    nbf.write(nb, nb_path)
    print("Notebook updated: accuracy and metrics cells appended.")
else:
    print("Notebook already contains the accuracy & metrics section; skipping append.")