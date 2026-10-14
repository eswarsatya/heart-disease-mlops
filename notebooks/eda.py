import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/heart.csv')
print(df.head())
print("Missing values:\n", df.isnull().sum())

# 1. Class Balance
plt.figure()
sns.countplot(x='target', data=df)
plt.title('Class Balance - Heart Disease')
plt.savefig('screenshots/class_balance.png')
print("Saved class_balance.png")

# 2. Histograms
df.hist(figsize=(12,10))
plt.tight_layout()
plt.savefig('screenshots/histograms.png')
print("Saved histograms.png")

# 3. Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('screenshots/heatmap.png')
print("Saved heatmap.png")

print("All EDA done!")