import pandas as pd

# Load the re-clustered data and the new transaction data
re_clustered_data_path = 'C:/Users/User/OneDrive/桌面/大數據競賽/練習/re_clustered_data.csv'
xxx2_data_path = 'C:/Users/User/OneDrive/桌面/大數據競賽/練習/XXX2.csv'

re_clustered_df = pd.read_csv(re_clustered_data_path)
xxx2_df = pd.read_csv(xxx2_data_path)

# Merge the transaction data with the re-clustered data based on customer
merged_df = pd.merge(xxx2_df, re_clustered_df[['customer', 'new_cluster']], on='customer', how='inner')

output_path = 'C:/Users/User/OneDrive/桌面/大數據競賽/練習/merged_transaction_data.csv'
merged_df.to_csv(output_path, index=False)

merged_data_path = 'C:/Users/User/OneDrive/桌面/大數據競賽/練習/merged_transaction_data.csv'
merged_df = pd.read_csv(merged_data_path)

# Convert the 'date' column to datetime format
merged_df['datetime'] = pd.to_datetime(merged_df['datetime'])

# Extract year and month
merged_df['year'] = merged_df['datetime'].dt.year
merged_df['month'] = merged_df['datetime'].dt.month

# Group by year, month, customer, and new_cluster, then aggregate the data
aggregated_df = merged_df.groupby(['year', 'month', 'customer', 'new_cluster']).agg(
    total_purchase_amount=('amount', 'sum'),
    total_transaction_count=('amount', 'count')
).reset_index()


# Save the resulting data to a new CSV file
output_path = 'C:/Users/User/OneDrive/桌面/大數據競賽/練習/merged_transaction_data.csv'
aggregated_df.to_csv(output_path, index=False)

# Display the first few rows of the merged data
aggregated_df.head()
