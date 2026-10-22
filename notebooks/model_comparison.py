# %% STEP 1: Imports
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, precision_recall_curve, average_precision_score)
from sklearn.inspection import permutation_importance

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Find the project root (the folder that contains data/health_lifestyle_classification.csv)
DATA_REL = Path("data/health_lifestyle_classification.csv")
start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next((p for p in [start, *start.parents] if (p / DATA_REL).exists()), None)
if ROOT is None:
    raise SystemExit("Could not find data/health_lifestyle_classification.csv")
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

# %% STEP 2: Load data (only EMPTY cells are missing; the text "None" is a real category)
df = pd.read_csv(ROOT / DATA_REL, keep_default_na=False, na_values=[""])
print(df.shape)
print(df["target"].value_counts())

# %% STEP 3: Prepare features and target
drop_cols = ["survey_code", "electrolyte_level", "environmental_risk_score"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

y = (df["target"] == "diseased").astype(int)      # 1 = diseased, 0 = healthy
X = df.drop(columns=["target"])
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
print("Numeric:", len(num_cols), "| Categorical:", len(cat_cols))

# %% STEP 4: Split 80/20 and preprocess
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("scale", MinMaxScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
])
Xtr = preprocess.fit_transform(X_train).astype("float32")
Xte = preprocess.transform(X_test).astype("float32")
print("Processed shapes:", Xtr.shape, Xte.shape)

# %% STEP 5: MODEL 1 - Recurrent Neural Network (Keras SimpleRNN)
Xtr_rnn = Xtr.reshape(Xtr.shape[0], 1, Xtr.shape[1])
Xte_rnn = Xte.reshape(Xte.shape[0], 1, Xte.shape[1])

rnn = keras.Sequential([
    keras.layers.Input(shape=(1, Xtr.shape[1])),
    keras.layers.SimpleRNN(64),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
rnn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy", metrics=["accuracy"])
rnn.summary()

history = rnn.fit(Xtr_rnn, y_train.values, epochs=20, batch_size=256,
                  validation_split=0.1, verbose=1)

# %% STEP 6: MODEL 2 - Gradient-boosted decision trees
gbt = HistGradientBoostingClassifier(learning_rate=0.08, max_iter=500,
                                     early_stopping=True, random_state=SEED)
gbt.fit(Xtr, y_train)
print("Trees used:", gbt.n_iter_)

# %% STEP 7: Predictions from both models
proba = {
    "RNN": rnn.predict(Xte_rnn, verbose=0).ravel(),
    "Gradient Boosting": gbt.predict_proba(Xte)[:, 1],
}

# %% STEP 8: Compare the two models (threshold 0.5)
rows = []
for name, p in proba.items():
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "Specificity": tn / (tn + fp),
        "ROC AUC": roc_auc_score(y_test, p),
        "PR AUC": average_precision_score(y_test, p),
    })
comparison = pd.DataFrame(rows).set_index("Model").round(4)
print(comparison)
comparison.to_csv(DOCS / "two_model_comparison.csv")

# %% STEP 9: Comparison plots
fig, ax = plt.subplots(2, 3, figsize=(17, 9))

for i, (name, p) in enumerate(proba.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test, (p >= 0.5).astype(int)),
                           display_labels=["healthy", "diseased"]).plot(ax=ax[0, i], colorbar=False)
    ax[0, i].set_title(f"Confusion Matrix: {name}")

for name, p in proba.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    ax[0, 2].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, p):.3f})")
ax[0, 2].plot([0, 1], [0, 1], "k--", label="Random")
ax[0, 2].set(title="ROC Curve", xlabel="False positive rate", ylabel="True positive rate")
ax[0, 2].legend()

for name, p in proba.items():
    pr, rc, _ = precision_recall_curve(y_test, p)
    ax[1, 0].plot(rc, pr, label=name)
ax[1, 0].axhline(y_test.mean(), color="k", ls="--", label="Baseline")
ax[1, 0].set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
ax[1, 0].legend()

ax[1, 1].plot(history.history["loss"], label="train loss")
ax[1, 1].plot(history.history["val_loss"], label="validation loss")
ax[1, 1].set(title="RNN training curve", xlabel="Epoch", ylabel="Loss")
ax[1, 1].legend()

for name, p in proba.items():
    ax[1, 2].hist(p, bins=40, alpha=0.5, label=name)
ax[1, 2].set(title="Predicted probability of disease", xlabel="Probability", ylabel="Count")
ax[1, 2].legend()

plt.tight_layout()
plt.savefig(DOCS / "two_model_comparison.png", dpi=150)
plt.show()

# %% STEP 10: Feature importance for the tree model
gbt_full = Pipeline([("prep", preprocess),
                     ("model", HistGradientBoostingClassifier(learning_rate=0.08, max_iter=500,
                                                              early_stopping=True,
                                                              random_state=SEED))])
gbt_full.fit(X_train, y_train)
imp = permutation_importance(gbt_full, X_test, y_test, scoring="roc_auc",
                             n_repeats=5, random_state=SEED, n_jobs=1)
imp_df = (pd.DataFrame({"feature": X_test.columns, "importance": imp.importances_mean})
          .sort_values("importance", ascending=False))
print(imp_df.head(15))

imp_df.head(15).iloc[::-1].plot.barh(x="feature", y="importance", legend=False, figsize=(8, 6))
plt.title("Top 15 features (Gradient Boosting)")
plt.tight_layout()
plt.savefig(DOCS / "feature_importance.png", dpi=150)
plt.show()

# %% STEP 11: Written conclusion
best = comparison["ROC AUC"].idxmax()
print(f"Best ROC AUC: {best} ({comparison.loc[best, 'ROC AUC']})")
if comparison["ROC AUC"].max() < 0.55:
    print("Both models are close to random guessing (AUC ~ 0.5): the features do not "
          "separate healthy from diseased individuals in this dataset.")