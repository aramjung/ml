import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# take titanic dataset and prepare input data 
def prepare(df):
        
    # Select relevant features  - TODO: feature engineering / ticket, cabin and title 
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    # select relevant columns into df (features + survived)
    # Handle missing values
    age_imputer = SimpleImputer(strategy="most_frequent")
    df["Age"] = age_imputer.fit_transform(df[["Age"]])
    # - best practise use a separate imputer because age and embarked are different data types 
    # - nan causes an error for the imputer.  must convert to string from object type first / nan becomes a string
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Embarked"] = df["Embarked"].astype("object")
    print(df["Embarked"].unique())

    # Encode categorical variables - assign number in alphabetical order
    label_enc = LabelEncoder()
    df["Sex"] = label_enc.fit_transform(df["Sex"])
    df["Embarked"] = label_enc.fit_transform(df["Embarked"])

    return (df, features)


