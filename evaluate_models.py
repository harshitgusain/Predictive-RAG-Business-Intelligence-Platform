import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load Data
df = pd.read_csv("data/Global_Superstore2.csv", encoding='latin1')

print("📊 STARTING DATA SCIENCE MODEL EVALUATION CORE...\n")

# ---------------------------------------------------
# 1. EVALUATING THE K-MEANS CLUSTERING MODEL
# ---------------------------------------------------
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y', errors='coerce')
df_clean = df.dropna(subset=['Order Date'])
snapshot_date = df_clean['Order Date'].max() + pd.Timedelta(days=1)

rfm = df_clean.groupby('Customer Name').agg({
    'Order Date': lambda x: (snapshot_date - x.max()).days,
    'Order ID': 'nunique',
    'Sales': 'sum'
}).rename(columns={'Order Date': 'Recency', 'Order ID': 'Frequency', 'Sales': 'Monetary'}).reset_index()

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Run K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(rfm_scaled)

# Calculate Silhouette Score
sil_score = silhouette_score(rfm_scaled, clusters)
print(f"--- Unsupervised Learning Diagnostics ---")
print(f"✅ K-Means Segmentation Silhouette Score: {sil_score:.4f}")
print("   *Interpretation: A score > 0 means clusters are well-separated and mathematically distinct.*")
print(f"   *Cluster Distributions:* {pd.Series(clusters).value_counts().to_dict()}\n")


# ---------------------------------------------------
# 2. EVALUATING THE TIME-SERIES FORECAST MODEL
# ---------------------------------------------------
monthly_sales = df_clean.groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales'].sum().reset_index()
monthly_sales = monthly_sales.set_index('Order Date').asfreq('MS').ffill()

# Train-Test Split (Hold out the last 6 months for testing validation accuracy)
train_data = monthly_sales['Sales'].iloc[:-6]
test_data = monthly_sales['Sales'].iloc[-6:]

# Fit model on training sequence
model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12)
fitted_model = model.fit()

# Predict on validation horizon
predictions = fitted_model.forecast(steps=6)

# Calculate Accuracy Metrics
mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
rmse = np.sqrt(np.mean((test_data - predictions) ** 2))

print(f"--- Predictive Time-Series Analytics Diagnostics ---")
print(f"✅ Forecasting Validation Model RMSE: ${rmse:,.2f}")
print(f"✅ Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print("   *Interpretation: A low MAPE confirms excellent model generalization on unseen data patterns.*")