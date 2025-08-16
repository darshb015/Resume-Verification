import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from xgboost import XGBRegressor

# ---------------------------
# 1. Load dataset
# ---------------------------
big_mart_data = pd.read_csv('Train.csv')

# ---------------------------
# 2. Handle missing values
# ---------------------------
big_mart_data['Item_Weight'] = big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean())

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
# 4. Feature engineering
# ---------------------------
big_mart_data['Outlet_Age'] = 2025 - big_mart_data['Outlet_Establishment_Year']
visibility_avg = big_mart_data.groupby('Item_Identifier')['Item_Visibility'].transform('mean')
big_mart_data['Item_Visibility_MeanRatio'] = big_mart_data['Item_Visibility'] / visibility_avg

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
X = big_mart_data.drop(columns='Item_Outlet_Sales')
Y = big_mart_data['Item_Outlet_Sales']

# Log-transform target
Y_log = np.log1p(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_log, test_size=0.2, random_state=2)

# ---------------------------
# 7. Fast GridSearchCV tuning
# ---------------------------
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=2)

# Reduced parameter grid for faster execution
param_grid = {
    'n_estimators': [300, 500],
    'learning_rate': [0.05, 0.1],
    'max_depth': [6, 8],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring='r2',
    verbose=1
)

grid_search.fit(X_train, Y_train)

best_model = grid_search.best_estimator_
print("\nBest Parameters:", grid_search.best_params_)

# ---------------------------
# 8. Evaluate tuned model
# ---------------------------
# Training set evaluation
train_pred = np.expm1(best_model.predict(X_train))
y_train_actual = np.expm1(Y_train)
r2_train = metrics.r2_score(y_train_actual, train_pred)

# Test set evaluation
test_pred = np.expm1(best_model.predict(X_test))
y_test_actual = np.expm1(Y_test)
r2_test = metrics.r2_score(y_test_actual, test_pred)

print(f"R² (Train): {r2_train:.4f}")
print(f"R² (Test) : {r2_test:.4f}")
