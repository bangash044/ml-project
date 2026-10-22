"""Train the deployable model and export backend/models/model_bundle.joblib.
Run from the project root:  python backend/train_export.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

# "None" is a real answer in some columns, so only EMPTY cells are missing.
df = pd.read_csv(ROOT / "data" / "health_lifestyle_classification.csv",
                 keep_default_na=False, na_values=[""])
print("Loaded:", df.shape)

drop_cols = ["survey_code", "electrolyte_level", "environmental_risk_score"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

y = (df["target"] == "diseased").astype(int)          # 1 = diseased, 0 = healthy
X = df.drop(columns=["target"])
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
print("Input features:", X.shape[1], "| numeric:", len(num_cols), "| categorical:", len(cat_cols))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("scale", MinMaxScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
])
pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", HistGradientBoostingClassifier(learning_rate=0.08, max_iter=500,
                                             early_stopping=True, random_state=SEED)),
])
pipe.fit(X_train, y_train)

proba = pipe.predict_proba(X_test)[:, 1]
print("Test accuracy:", round(accuracy_score(y_test, proba >= 0.5), 4))
print("Test ROC AUC :", round(roc_auc_score(y_test, proba), 4))

bundle = {
    "pipeline": pipe,
    "features": list(X_train.columns),     # exact training input order
    "task": "classification",
    "labels": ["healthy", "diseased"],     # index 0 / 1
    "model_version": "1.0",
}
(ROOT / "backend" / "models").mkdir(parents=True, exist_ok=True)
joblib.dump(bundle, ROOT / "backend" / "models" / "model_bundle.joblib")

# Sample = first COMPLETE test row (no blanks), so the form can be filled in fully.
sample = X_test.dropna().head(1)
sample.to_csv(ROOT / "data" / "sample_inputs.csv", index=False)
print("Saved bundle with", len(bundle["features"]), "features; sample row index:", sample.index[0])
print("Model's prediction for the sample:", pipe.predict_proba(sample)[0].round(4).tolist())