# Umemployment-in-india
#umemployment in india project for data science 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
uploaded = files.upload()

df = pd.read_csv("Unemployment in India.csv")
df.columns = df.columns.str.strip()
print("First few rows of the dataset:")
print(df.head())
print("\nMissing values in the dataset:")
print(df.isnull().sum())
df = df.dropna()
df['Date'] = pd.to_datetime(df['Date'])
print("\nBasic statistics of the dataset:")
print(df.describe())
plt.figure(figsize=(14, 7))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=df)
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 5))
sns.histplot(df['Estimated Unemployment Rate (%)'], kde=True)
plt.title('Distribution of Unemployment Rate')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Frequency')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df.set_index('Date', inplace=True)
X = np.array(range(len(df))).reshape(-1, 1)  # Time as a feature
y = df['Estimated Unemployment Rate (%)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Estimated Unemployment Rate (%)'], label='Actual')
plt.plot(df.index[X_test.flatten()], y_pred, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Unemployment Rate')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.grid(True)
plt.show()
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
