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

# 按退貨總數降序排列，並選擇前3名退貨最多的產品
top3_products = product_return_counts.sort_values(by='return_count', ascending=False).head(3)
top3_product_names = top3_products['product'].tolist()

# 選擇特定客戶的退貨產品
customer_ids = ['c5871', 'c2', 'c9']
customer_returns = data[(data['is_return']) & (data['customer'].isin(customer_ids)) & (data['product'].isin(top3_product_names))]

# 計算每個客戶退貨的產品及其次數
customer_return_products = customer_returns.groupby(['customer', 'product']).size().reset_index(name='return_count')

# 繪製退貨產品的柱狀圖
fig, ax = plt.subplots(figsize=(12, 6))

# 獲取唯一產品列表
products = top3_product_names

# 設置每個客戶的顏色
colors = ['skyblue', 'salmon', 'lightgreen']
width = 0.25

# 為每個客戶繪製柱狀圖
for i, customer in enumerate(customer_ids):
    customer_data = customer_return_products[customer_return_products['customer'] == customer]
    x_positions = [products.index(product) + width*i for product in customer_data['product']]
    ax.bar(x_positions, customer_data['return_count'], width=width, color=colors[i], align='center', label=customer)

# 設置X軸標籤
ax.set_xticks(range(len(products)))
ax.set_xticklabels(products, rotation=90)

# 設置標題和標籤
ax.set_xlabel('Product')
ax.set_ylabel('Return Count')
ax.set_title('Return Count of Top 3 Returned Products for Selected Customers')
ax.legend(title='Customer')
plt.grid(axis='y')
plt.tight_layout()
plt.show()