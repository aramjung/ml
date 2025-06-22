import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import titanic_cleanup as cleanup
import joblib

# Load the dataset
df = pd.read_csv("../data/train.csv")
y = df["Survived"]

(df, features) = cleanup.prepare(df)

# Split data into train, validation, and test sets
X = df[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestClassifier(random_state=42)

# Set up GridSearchCV parameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 5-fold cross-validation
grid_search_cv = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_cv.fit(X_train, y_train)
best_model_cv = grid_search_cv.best_estimator_
y_test_pred = best_model_cv.predict(X_test)
accuracy_cv = accuracy_score(y_test, y_test_pred)
print(f"5-Fold Cross-Validation Accuracy: {accuracy_cv:.4f}")

# print(f"Best Parameters (Single Validation Set): {grid_search_single.best_params_}")
print(f"Best Parameters (5-Fold CV): {grid_search_cv.best_params_}")

joblib.dump(best_model_cv, "../model/random_forest.pkl")
