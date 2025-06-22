import pandas as pd
import transform as transform 
from sklearn.ensemble import RandomForestRegressor
import joblib

#load data
df = pd.read_csv('../data/test.csv')

def printDF(df, step):
    print(f'step {step}: number of columns {len(df.columns)}')
    print(f'columns {df.columns}')
X = df.copy()
X.pop('Id')

printDF(X, 0)
# 1. impute data with averages and modes from training data
X = transform.imput_transform(X)
printDF(X, 1)
# 2. encode 
X = transform.encode_transform(X, True)
printDF(X, 2)
# 3. cluster
features_to_cluster = [
    'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'LotFrontage', 'LotArea', 'MasVnrArea',
    'BsmtFinSF1', 'BsmtFinSF2', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GarageArea',
    'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'
]

X = transform.cluster_transform(X, features_to_cluster, 7)
printDF(X, 3)
# 4. mutual analysis
X = transform.mutual_information_transform(X)
printDF(X, 4)

print(f'number of columns {len(X.columns)}')
print(f'columns {X.columns}')

features = transform.get_features()

model = joblib.load('../Model/random_forest.pkl')
y_pred = model.predict(X[features])

result = pd.concat([df['Id'], pd.Series(y_pred)], axis=1)
result.rename(columns={1:'SalePrice'})
result.to_csv('../data/output.csv', index=False)
print('complete')