import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# 加载数据
file_path = 'C:/Users/User/OneDrive/桌面/大數據競賽/練習/XXX2.csv'
data = pd.read_csv(file_path)

# 确保日期列为日期时间格式
data['datetime'] = pd.to_datetime(data['datetime'])

# 创建新的时间特征
data['month'] = data['datetime'].dt.month
data['quarter'] = data['datetime'].dt.quarter

# 计算每月的交易数量和新增特征
monthly_data = data.groupby(data['datetime'].dt.to_period('M')).agg(
    transaction_count=('amount', 'count'),
    total_sales=('amount', 'sum'),
    avg_transaction_amount=('amount', 'mean')
).reset_index()
monthly_data['datetime'] = monthly_data['datetime'].dt.to_timestamp()
monthly_data['month'] = monthly_data['datetime'].dt.month
monthly_data['quarter'] = monthly_data['datetime'].dt.quarter

# 按日期排序
monthly_data = monthly_data.sort_values('datetime')

# 将数据分为训练集和测试集
train_data = monthly_data[monthly_data['datetime'] < '2017-02-01']
test_data = monthly_data[monthly_data['datetime'] >= '2017-02-01']

X_train = train_data[['total_sales', 'avg_transaction_amount', 'month', 'quarter']]
y_train = train_data['transaction_count']

X_test = test_data[['total_sales', 'avg_transaction_amount', 'month', 'quarter']]
y_test = test_data['transaction_count']

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用XGBoost回归模型
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train_scaled, y_train)

# 进行预测
y_pred = model.predict(X_test_scaled)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R^2 Score: {r2}')

# 获取测试集的实际日期
test_dates = test_data['datetime']

# 确保测试集按照日期排序
test_results = pd.DataFrame({'date': test_dates, 'actual': y_test.values, 'predicted': y_pred})
test_results = test_results.sort_values('date')

# 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(test_results['date'], test_results['actual'], label='Actual', marker='o')
plt.plot(test_results['date'], test_results['predicted'], label='Predicted', marker='x')
plt.xlabel('Date')
plt.ylabel('Transaction Count')
plt.title('Actual vs Predicted Transaction Count')
plt.legend()
plt.grid(True)
# plt.savefig('C:/Users/User/OneDrive/桌面/大數據競賽/練習/圖表/Actual_vs_Predicted_Transaction_Count_XGB.png')
plt.show()

# 进行预测
y_pred = model.predict(X_train_scaled)

# 评估模型性能
mse = mean_squared_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R^2 Score: {r2}')