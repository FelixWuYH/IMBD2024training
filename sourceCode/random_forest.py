import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 加載上傳的 CSV 文件
file_path = 'C:/Users/USER/OneDrive/桌面/ML/處理過.csv'
data = pd.read_csv(file_path)

# 轉換日期列為日期類型
data['datetime'] = pd.to_datetime(data['datetime'])

# 計算每筆交易的銷售額
data['amount'] = data['price'] * data['quantity']

# 過濾2017/07之後的數據
filtered_data = data[data['datetime'] < '2017-07-01']

# 按月分組，計算每月的總銷售額
monthly_sales = filtered_data.resample('MS', on='datetime')['amount'].sum().reset_index()

# 創建時間特徵
monthly_sales['month'] = monthly_sales['datetime'].dt.month
monthly_sales['year'] = monthly_sales['datetime'].dt.year

# 訓練集和測試集
train_data = monthly_sales[monthly_sales['datetime'] < '2017-01-01']
test_data = monthly_sales[monthly_sales['datetime'] >= '2017-01-01']

# 特徵和目標變量
X_train = train_data[['month', 'year']]
y_train = train_data['amount']
X_test = test_data[['month', 'year']]
y_test = test_data['amount']

# 建立隨機森林回歸模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 評估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# 繪製結果
plt.figure(figsize=(12, 6))
plt.plot(train_data['datetime'], y_train, label='Historical Sales (Train)')
plt.plot(test_data['datetime'], y_test, label='Historical Sales (Test)')
plt.plot(test_data['datetime'], y_pred, color='red', linestyle='--', label='Forecasted Sales')
plt.title('Monthly Sales Forecast with Random Forest')
plt.xlabel('Month')
plt.ylabel('Total Sales Amount')
plt.legend()
plt.grid(True)
plt.show()

# 預測未來12個月
future_dates = pd.date_range(start='2017-07-01', periods=12, freq='MS')
future_features = pd.DataFrame({
    'month': future_dates.month,
    'year': future_dates.year
})
future_pred = model.predict(future_features)

# 繪製未來預測結果
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['datetime'], monthly_sales['amount'], label='Historical Sales')
plt.plot(future_dates, future_pred, color='red', linestyle='--', label='Forecasted Sales')
plt.title('Future Monthly Sales Forecast with Random Forest')
plt.xlabel('Month')
plt.ylabel('Total Sales Amount')
plt.legend()
plt.grid(True)
plt.show()