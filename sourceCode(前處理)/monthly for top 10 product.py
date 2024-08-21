import pandas as pd
import matplotlib.pyplot as plt

# 加載上傳的 CSV 文件
file_path = 'C:/Users/USER/OneDrive/桌面/ML/XXX2.csv'
data = pd.read_csv(file_path)

# 轉換日期列為日期類型
data['datetime'] = pd.to_datetime(data['datetime'])

# 標記退貨交易
data['is_return'] = data['quantity'] < 0

# 計算每個產品的退貨總數
product_return_counts = data[data['is_return']].groupby('product').size().reset_index(name='return_count')

# 按退貨總數降序排列，並選擇前10名退貨最多的產品
top10_products = product_return_counts.sort_values(by='return_count', ascending=False).head(10)

# 顯示前10名退貨產品
print("前10名退貨產品：")
print(top10_products)

# 繪製退貨總數前10名產品的柱狀圖
plt.figure(figsize=(12, 6))
plt.bar(top10_products['product'], top10_products['return_count'], color='salmon')
plt.xlabel('Product')
plt.ylabel('Return Count')
plt.title('Top 10 Returned Products')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 統計每個產品的退貨分佈（按月）
monthly_returns = data[data['is_return']].groupby(['product', data['datetime'].dt.to_period('M')]).size().reset_index(name='return_count')
monthly_returns['datetime'] = monthly_returns['datetime'].dt.to_timestamp()

# 選擇前10名產品的數據
top10_monthly_returns = monthly_returns[monthly_returns['product'].isin(top10_products['product'])]

# 繪製前10名產品的每月退貨數量
plt.figure(figsize=(14, 8))
for product in top10_products['product']:
    product_data = top10_monthly_returns[top10_monthly_returns['product'] == product]
    plt.plot(product_data['datetime'], product_data['return_count'], label=product)

plt.title('Monthly Returns for Top 10 Products')
plt.xlabel('Date')
plt.ylabel('Return Count')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()