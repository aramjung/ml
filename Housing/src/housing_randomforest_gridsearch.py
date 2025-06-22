import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import transform as transform 
import joblib

# load data
df = pd.read_csv("../data/train.csv")
y = df.pop('SalePrice')
df.pop('Id')

def printDF(df, step):
    print(f'step {step}: number of columns {len(df.columns)}')
    print(f'columns {df.columns}')

printDF(df, 0)
# 1. impute 
X = transform.impute_fit_transform(df)
printDF(X, 1)
# 2. encode / use label encoding when using decision tree based models 
X = transform.encode_fit_transform(X, True)
printDF(X, 2)
# 3. cluster 
features_to_cluster = [
    'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'LotFrontage', 'LotArea', 'MasVnrArea',
    'BsmtFinSF1', 'BsmtFinSF2', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GarageArea',
    'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'
]
X = transform.cluster_fit_transform(X, features_to_cluster, 7)
printDF(X, 3)
# 4. mutual analysis to drop low scoring features and create high scoring interaction features 
X = transform.mutual_information_fit_transform(X, y)
printDF(X, 4)
# Split data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'number of columns {len(X.columns)}')
print(f'columns {X.columns}')

# Define the model
# bug RandomForestRegressor must be used instead because y values are continous  
model = RandomForestRegressor(random_state=42)

# Set up GridSearchCV parameters
param_grid = {
    'n_estimators': [100],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2],
}

# 5-fold cross-validation   cv=5 / used 3 fold instead 
grid_search_cv = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search_cv.fit(X_train, y_train)
best_model_cv = grid_search_cv.best_estimator_
# Predict on test set
y_test_pred = best_model_cv.predict(X_test)
# Evaluate using R-squared and MSE
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
print(f"5-Fold Cross-Validation Test MSE: {mse:.2f}")
print(f"5-Fold Cross-Validation Test R^2: {r2:.4f}")

# print(f"Best Parameters (Single Validation Set): {grid_search_single.best_params_}")
print(f"Best Parameters (5-Fold CV): {grid_search_cv.best_params_}")

joblib.dump(best_model_cv, '../Model/random_forest.pkl')

transform.save_features(X.columns)
print('complete')
