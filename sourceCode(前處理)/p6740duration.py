import pandas as pd

# 加載上傳的 CSV 文件
file_path = 'C:/Users/USER/OneDrive/桌面/ML/XXX2.csv'
data = pd.read_csv(file_path)

# 轉換日期列為日期類型
data['datetime'] = pd.to_datetime(data['datetime'])

# 篩選出客戶 c5871 的數據
customer_id = 'c5871'
customer_data = data[data['customer'] == customer_id]

# 獲取客戶 c5871 出現的時間範圍
start_date = customer_data['datetime'].min()
end_date = customer_data['datetime'].max()

print(f"客戶 {customer_id} 出現的時間範圍：從 {start_date} 到 {end_date}")