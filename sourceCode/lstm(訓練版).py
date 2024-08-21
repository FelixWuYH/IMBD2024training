import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
file_path = 'C:/Users/User/OneDrive/桌面/大數據競賽/練習/XXX2.csv'
data = pd.read_csv(file_path)
# Convert datetime to date
data['datetime'] = pd.to_datetime(data['datetime'])
data['date'] = data['datetime'].dt.date

# Aggregate data by date to get daily sales
daily_sales = data.groupby('date').agg({'quantity': 'sum'}).reset_index()
daily_sales['date'] = pd.to_datetime(daily_sales['date'])
daily_sales.set_index('date', inplace=True)

# Split the data into training and validation sets
train_data = daily_sales[:'2017-07-31']
val_data = daily_sales # ['2017-08-01':]

# Standardize the data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
val_scaled = scaler.transform(val_data)

# Create sequences for LSTM
def create_sequences(data, n_past):
    X, Y = [], []
    for i in range(len(data) - n_past):
        X.append(data[i:i + n_past])
        Y.append(data[i + n_past])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

n_past = 90  # Use past 30 days to predict the next day
X_train, Y_train = create_sequences(train_scaled, n_past)
X_val, Y_val = create_sequences(val_scaled, n_past)

# Create DataLoader
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

# Define LSTM model
class LSTMSales(nn.Module):
    def __init__(self, n_features, hidden_dim, n_layers=2):
        super(LSTMSales, self).__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return x

n_features = 1
hidden_dim = 64

model = LSTMSales(n_features, hidden_dim)
model.to('cuda')

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
losses = []
val_losses = []
# Training the model
epochs = 20

for epoch in range(epochs):
    model.train()
    epoch_loss = 0  # 初始化每個 epoch 的損失
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to('cuda'), y_batch.to('cuda')
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()  # 累計 batch 損失
        
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to('cuda'), y_val.to('cuda')
            y_pred = model(x_val)
            val_loss += criterion(y_pred, y_val).item()

    val_losses.append(val_loss/ len(val_loader))
    # print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    losses.append(epoch_loss / len(train_loader))  # 記錄每個 epoch 的平均損失

# 繪製損失下降過程圖
# plt.plot(range(epochs), val_losses)
plt.plot(range(len(losses)), losses, label='Train Loss')
plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
# plt.plot(range(len(val_losses)), val_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.show()

# Validation
model.eval()
predictions, targets = [], []
with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch, y_batch = x_batch.to('cuda'), y_batch.to('cuda')
        y_pred = model(x_batch)
        predictions.append(y_pred.cpu().numpy())
        targets.append(y_batch.cpu().numpy())

predictions = np.concatenate(predictions)
targets = np.concatenate(targets)

# Inverse transform the predictions and targets
predictions = scaler.inverse_transform(predictions)
targets = scaler.inverse_transform(targets)

# 只顯示2017-08-01之後的數據
start_date = '2017-08-01'
filtered_dates = daily_sales.index[-len(targets):]
filtered_dates = filtered_dates[filtered_dates >= start_date]

filtered_predictions = predictions[-len(filtered_dates):]
filtered_targets = targets[-len(filtered_dates):]

# Calculate MAE and MAPE
mae = np.mean(np.abs(predictions - targets))
mape = np.mean(np.abs((predictions - targets) / targets)) * 100

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(filtered_dates, filtered_targets, label='True Sales')
plt.plot(filtered_dates, filtered_predictions, label='Predicted Sales')
plt.title(f'MAE: {mae:.2f}, MAPE: {mape:.2f}%')
plt.legend()
plt.show()

# Define file path to save the model
# model_save_path = 'C:/Users/User/OneDrive/桌面/大數據競賽/練習/模型庫/lstm_sales_model(20).pth'

# Save the model
# torch.save(model.state_dict(), model_save_path)