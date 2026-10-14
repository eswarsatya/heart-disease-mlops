import pandas as pd, os
os.makedirs("data", exist_ok=True)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
df = pd.read_csv(url, names=cols)
df.replace('?', pd.NA, inplace=True)
df.to_csv("data/heart.csv", index=False)
print(f"Saved {df.shape}")