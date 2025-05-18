
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore


df = pd.read_csv('data/benin.csv')  


print(df.describe())
print("\nMissing values:\n", df.isna().sum())


missing = df.isna().mean()
print("\nColumns with >5% missing:\n", missing[missing > 0.05])


cols_to_check = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust']
z_scores = df[cols_to_check].apply(zscore)
outliers = (np.abs(z_scores) > 3).any(axis=1)
print(f"\nOutliers count: {outliers.sum()}")

df_clean = df.copy()
df_clean[cols_to_check] = df_clean[cols_to_check].fillna(df_clean[cols_to_check].median())

df_clean.to_csv('data/benin_clean.csv', index=False)


df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

df[['GHI', 'DNI', 'DHI', 'Tamb']].plot(figsize=(15, 5))
plt.title("GHI, DNI, DHI, and Tamb over Time")
plt.ylabel("Value")
plt.xlabel("Time")
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(df_clean[['GHI', 'DNI', 'DHI', 'ModA', 'ModB']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.scatterplot(x='WS', y='GHI', data=df_clean)
plt.title("WS vs GHI")
plt.show()

df_clean['GHI'].hist(bins=30)
plt.title("Histogram of GHI")
plt.xlabel("GHI")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(df_clean['GHI'], df_clean['Tamb'], s=df_clean['RH'], alpha=0.5)
plt.title("GHI vs Tamb with RH as bubble size")
plt.xlabel("GHI")
plt.ylabel("Tamb")
plt.grid(True)
plt.show()
