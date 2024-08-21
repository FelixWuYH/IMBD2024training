import pandas as pd
import matplotlib.pyplot as plt

# 加載上傳的 CSV 文件
file_path = 'C:/Users/USER/OneDrive/桌面/ML/XXX2.csv'
data = pd.read_csv(file_path)

# 轉換日期列為日期類型
data['datetime'] = pd.to_datetime(data['datetime'])

# 標記退貨交易
data['is_return'] = data['quantity'] < 0

# 計算每個客戶的退貨總數
customer_return_counts = data[data['is_return']].groupby('customer').size().reset_index(name='return_count')

# 顯示前10名退貨次數最多的客戶
top10_customers = customer_return_counts.sort_values(by='return_count', ascending=False).head(10)
print("前10名退貨次數最多的客戶：")
print(top10_customers)

# 繪製退貨次數前10名客戶的柱狀圖
plt.figure(figsize=(12, 6))
plt.bar(top10_customers['customer'], top10_customers['return_count'], color='skyblue')
plt.xlabel('Customer')
plt.ylabel('Return Count')
plt.title('Top 10 Customers by Return Count')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 繪製所有客戶的退貨次數分佈
plt.figure(figsize=(12, 6))
plt.hist(customer_return_counts['return_count'], bins=30, color='salmon', edgecolor='black')
plt.xlabel('Return Count')
plt.ylabel('Number of Customers')
plt.title('Distribution of Return Counts by Customer')
plt.grid(axis='y')
plt.tight_layout()
plt.show()