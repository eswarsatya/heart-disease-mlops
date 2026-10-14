import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def get_preprocessor():
    num_cols = ['age','trestbps','chol','thalach','oldpeak']
    preprocessor = ColumnTransformer([
        ('scaler', StandardScaler(), num_cols)
    ], remainder='passthrough')
    return preprocessor

def load_data():
    df = pd.read_csv("data/heart.csv")
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)
    df = df.astype({c: float for c in df.columns})
    X = df.drop('target', axis=1)
    y = df['target'].apply(lambda x: 1 if x>0 else 0)
    return X, y