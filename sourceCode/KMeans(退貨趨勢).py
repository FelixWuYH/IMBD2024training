import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 讀取數據
file_path = 'C:/Users/USER/OneDrive/桌面/ML/XXX2.csv'
data = pd.read_csv(file_path)

# 檢查數據框的前幾行和列名
print("數據框的前幾行:")
print(data.head())
print("數據框的列名:")
print(data.columns)

# 過濾退貨數據（假設 amount 列表示金額）
data['is_return'] = data['amount'] < 0

# 確保日期列為日期類型並按日期排序
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.sort_values('datetime')

# 計算每月的退貨情況和新增特徵
monthly_returns = data.groupby(data['datetime'].dt.to_period('M')).agg(
    is_return=('is_return', 'sum'),
    total_sales=('amount', 'sum'),
    transaction_count=('amount', 'count'),
    avg_transaction_amount=('amount', 'mean')
).reset_index()
monthly_returns['datetime'] = monthly_returns['datetime'].dt.to_timestamp()
monthly_returns['is_return'] = monthly_returns['is_return'] > 0  # 將退貨金額轉換為是否發生退貨
monthly_returns['month'] = monthly_returns['datetime'].dt.month
monthly_returns['quarter'] = monthly_returns['datetime'].dt.quarter

# 打印計算出的每月退貨情況和新增特徵
print("\n每月退貨情況和新增特徵:")
print(monthly_returns.head())

# 創建滯後特徵
def create_features(df, target_column, window=6):
    for i in range(1, window + 1):
        df[f'lag_{target_column}_{i}'] = df[target_column].shift(i)
    df = df.dropna()
    return df

# 創建滯後特徵
for col in ['is_return', 'total_sales', 'transaction_count', 'avg_transaction_amount']:
    monthly_returns = create_features(monthly_returns, col)

# 打印創建滯後特徵後的數據
print("\n創建滯後特徵後的數據:")
print(monthly_returns.head())

# 定義特徵
feature_columns = [f'lag_is_return_{i}' for i in range(1, 7)] + \
                  [f'lag_total_sales_{i}' for i in range(1, 7)] + \
                  [f'lag_transaction_count_{i}' for i in range(1, 7)] + \
                  [f'lag_avg_transaction_amount_{i}' for i in range(1, 7)] + \
                  ['month', 'quarter']

X = monthly_returns[feature_columns]

# 標準化特徵
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 嘗試不同的聚類數量
sil_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    sil_scores.append((n_clusters, sil_score))

# 打印每個聚類數量的 Silhouette Score
print("\n不同聚類數量的 Silhouette Scores:")
for n_clusters, sil_score in sil_scores:
    print(f"聚類數量: {n_clusters}, Silhouette Score: {sil_score}")

# 選擇最佳的聚類數量
best_n_clusters = max(sil_scores, key=lambda x: x[1])[0]
print(f"\n最佳的聚類數量: {best_n_clusters}")

# 使用最佳的聚類數量進行聚類
kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
kmeans.fit(X_scaled)
monthly_returns['cluster'] = kmeans.labels_

# 打印聚類結果
print("\n聚類結果:")
print(monthly_returns[['datetime', 'cluster']])

# 繪製每月的退貨趨勢圖
plt.figure(figsize=(12, 6))
for cluster in monthly_returns['cluster'].unique():
    cluster_data = monthly_returns[monthly_returns['cluster'] == cluster]
    plt.plot(cluster_data['datetime'], cluster_data['total_sales'], marker='o', linestyle='-', label=f'Cluster {cluster}')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.title('Monthly Total Sales by Cluster')
plt.legend()
plt.grid(True)
plt.show()

# 繪製不同聚類數量的 Silhouette Scores
n_clusters_list, sil_scores_list = zip(*sil_scores)
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_list, sil_scores_list, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different Number of Clusters')
plt.grid(True)
plt.show()
