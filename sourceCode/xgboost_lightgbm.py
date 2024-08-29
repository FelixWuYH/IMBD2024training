from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 讀取數據
file_path = 'C:/Users/User/OneDrive/桌面/大數據競賽/練習/XXX2.csv'
data = pd.read_csv(file_path)

# 轉換 datetime 列為日期時間格式
data['datetime'] = pd.to_datetime(data['datetime'])

# 刪除原始 datetime 和 invoiceNo 列
data.drop(columns=['datetime', 'invoiceNo'], inplace=True)

# 創建新特徵：price * quantity
data['price_quantity'] = data['price'] * data['quantity']

# One-hot 編碼 'channel' 類別變量
data_encoded = pd.get_dummies(data, columns=['channel'])

# 刪除 amount 小於 0 的數據
data_filtered = data_encoded[data_encoded['amount'] >= 0]

# 定義特徵和目標變量
X = data_filtered[['price_quantity'] + [col for col in data_filtered.columns if 'channel' in col]]
y = data_filtered['amount']

# 將數據分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化並訓練 LightGBM 模型
lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_model.fit(X_train, y_train)

# 使用 LightGBM 進行預測
y_pred_lgb = lgb_model.predict(X_test)

# 初始化並訓練 XGBoost 模型
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# 使用 XGBoost 進行預測
y_pred_xgb = xgb_model.predict(X_test)

# 計算評估指標
metrics = {
    'Model': ['LightGBM', 'XGBoost'],
    'RMSE': [np.sqrt(mean_squared_error(y_test, y_pred_lgb)), np.sqrt(mean_squared_error(y_test, y_pred_xgb))],
    'MAE': [mean_absolute_error(y_test, y_pred_lgb), mean_absolute_error(y_test, y_pred_xgb)],
    'R²': [r2_score(y_test, y_pred_lgb), r2_score(y_test, y_pred_xgb)]
}

metrics_df = pd.DataFrame(metrics)

# 繪製預測值與實際值的比較圖
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lgb, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('LightGBM: Actual vs Predicted')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_xgb, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('XGBoost: Actual vs Predicted')

plt.tight_layout()
plt.show()

mask = y_test <= 15000
y_test_filtered = y_test[mask]
y_pred_lgb_filtered = y_pred_lgb[mask]
y_pred_xgb_filtered = y_pred_xgb[mask]

# 繪製預測值與實際值的比較圖
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test_filtered, y_pred_lgb_filtered, color='green')
plt.plot([y_test_filtered.min(), y_test_filtered.max()], [y_test_filtered.min(), y_test_filtered.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('LightGBM: Actual vs Predicted (amount <= 15000)')

plt.subplot(1, 2, 2)
plt.scatter(y_test_filtered, y_pred_xgb_filtered, color='blue')
plt.plot([y_test_filtered.min(), y_test_filtered.max()], [y_test_filtered.min(), y_test_filtered.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('XGBoost: Actual vs Predicted (amount <= 15000)')

plt.tight_layout()
plt.show()