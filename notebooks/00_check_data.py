import pandas as pd

df = pd.read_csv("data/health_lifestyle_classification.csv",
                 keep_default_na=False, na_values=[""])
print("Shape:", df.shape)                      # expect about (100000, 48)
print(df["target"].value_counts())
print("Missing per column (top 10):")
print(df.isna().sum().sort_values(ascending=False).head(10))
print("Columns:", list(df.columns))