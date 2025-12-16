from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load dataset and standardize column names like in the pipeline
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

data = load_breast_cancer()
features_df = pd.DataFrame(data.data, columns=data.feature_names).rename(columns=renamed)
# Export first 15 rows without labels; this is our unlabeled input
sample = features_df.iloc[:15].copy()
Path("data").mkdir(parents=True, exist_ok=True)
sample.to_csv("data/sample_unlabeled.csv", index=False)
print("Saved: data/sample_unlabeled.csv with", sample.shape[0], "rows and", sample.shape[1], "features")