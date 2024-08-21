import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree

file_path = 'XXX2_cleaned.csv'
data = pd.read_csv(file_path)

data.dropna()

data.dropna(subset=['cost'], inplace=True)

missing_values = data.isnull().sum()
missing_values

label_encoders = {}
for column in ['channel', 'customer', 'product', 'category', 'category2']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# 選擇特徵和目標變量
features = ['channel', 'customer', 'product', 'category', 'price', 'amount', 'category2', 'cost']
target = 'quantity'  # 這裡假設我們要預測category，可以根據需求改變

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
