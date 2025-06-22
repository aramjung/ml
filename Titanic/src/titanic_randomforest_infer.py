import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import titanic_cleanup as cleanup
import joblib

df = pd.read_csv("../data/test.csv")

(df, features) = cleanup.prepare(df)

# Split data into train, validation, and test sets
X = df[features]

loaded_model = joblib.load("../model/random_forest.pkl")
y_pred = loaded_model.predict(X)

result = pd.concat([df["PassengerId"], pd.Series(y_pred)], axis=1)
result.rename(columns={1:'Survived'})
result.to_csv("../data/output.csv", index=False)
print("complete")