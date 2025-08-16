import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# ---------------------------
# 1. Load dataset
# ---------------------------
big_mart_data = pd.read_csv('Train.csv')

# Quick look at the data
print(big_mart_data.info())
print(big_mart_data.isnull().sum())

# ---------------------------
# 2. Handle missing values
# ---------------------------
big_mart_data.fillna({'Item_Weight': big_mart_data['Item_Weight'].mean()}, inplace=True)

# Fill Outlet_Size missing values using mode by Outlet_Type
mode_of_Outlet_size = big_mart_data.pivot_table(
    values='Outlet_Size',
    columns='Outlet_Type',
    aggfunc=lambda x: x.mode()[0]
)
miss_values = big_mart_data['Outlet_Size'].isnull()
big_mart_data.loc[miss_values, 'Outlet_Size'] = big_mart_data.loc[miss_values, 'Outlet_Type'].apply(
    lambda x: mode_of_Outlet_size[x]
)

print(big_mart_data.isnull().sum())  # verify no missing values remain

# ---------------------------
# 3. Data cleaning
# ---------------------------
big_mart_data.replace({
    'Item_Fat_Content': {
        'low fat': 'Low Fat',
        'LF': 'Low Fat',
        'reg': 'Regular'
    }
}, inplace=True)

# ---------------------------
# 4. Data visualization
# ---------------------------
plt.figure(figsize=(6, 6))
sns.histplot(big_mart_data['Item_Weight'], kde=True)
plt.show()

plt.figure(figsize=(6, 6))
sns.histplot(big_mart_data['Item_Visibility'], kde=True)
plt.show()

plt.figure(figsize=(6, 6))
sns.histplot(big_mart_data['Item_MRP'], kde=True)
plt.show()

plt.figure(figsize=(6, 6))
sns.histplot(big_mart_data['Item_Outlet_Sales'], kde=True)
plt.show()

plt.figure(figsize=(6, 6))
sns.countplot(x='Outlet_Establishment_Year', data=big_mart_data)
plt.show()

plt.figure(figsize=(6, 6))
sns.countplot(x='Item_Fat_Content', data=big_mart_data)
plt.show()

plt.figure(figsize=(30, 6))
sns.countplot(x='Item_Type', data=big_mart_data)
plt.show()

plt.figure(figsize=(6, 6))
sns.countplot(x='Outlet_Size', data=big_mart_data)
plt.show()

# ---------------------------
# 5. Encode categorical columns
# ---------------------------
encoder = LabelEncoder()
for col in big_mart_data.columns:
    if big_mart_data[col].dtype == 'object':
        big_mart_data[col] = encoder.fit_transform(big_mart_data[col])

# ---------------------------
# 6. Split features & target
# ---------------------------
X = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
Y = big_mart_data['Item_Outlet_Sales']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# ---------------------------
# 7. Train the model
# ---------------------------
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)

# ---------------------------
# 8. Evaluate the model
# ---------------------------
training_data_prediction = regressor.predict(X_train)
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R Squared value (Train) = ', r2_train)

test_data_prediction = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R Squared value (Test) = ', r2_test)
