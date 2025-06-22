import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('../data/train.csv')
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

selection = ['Age', 'Fare', 'Embarked', 'Sex', 'SibSp', 'Pclass', 'Parch']
X = data[selection]
y = data['Survived']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

max_degree = 10
train_accuracies, cv_accuracies, test_accuracies = [], [], []

for degree in range(1, max_degree + 1):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_cv = poly.transform(X_cv)
    X_poly_test = poly.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_poly_train)
    X_cv_scaled = scaler.transform(X_poly_cv)
    X_test_scaled = scaler.transform(X_poly_test)

    model = LogisticRegression(C=0.05, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_cv = model.predict(X_cv_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, y_pred_train)
    cv_acc = accuracy_score(y_cv, y_pred_cv)
    test_acc = accuracy_score(y_test, y_pred_test)

    train_accuracies.append(train_acc)
    cv_accuracies.append(cv_acc)
    test_accuracies.append(test_acc)

    print(f'Degree {degree}: Train Acc: {train_acc:.4f}, CV Acc: {cv_acc:.4f}, Test Acc: {test_acc:.4f}')

    plt.figure(figsize=(5, 4))
    # heapmap displays the dataframe or matrix 
    sns.heatmap(confusion_matrix(y_cv, y_pred_cv), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Degree {degree})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

plt.figure(figsize=(8, 5))
plt.plot(range(1, max_degree + 1), train_accuracies, marker='o', label='Train Accuracy')
plt.plot(range(1, max_degree + 1), cv_accuracies, marker='s', label='CV Accuracy')
plt.plot(range(1, max_degree + 1), test_accuracies, marker='^', label='Test Accuracy')
plt.xlabel('Polynomial Degree')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Polynomial Degree')
plt.legend()
plt.grid()
plt.show()




#print(data.describe)
# preprocess data 
# deal with not available - some cabins are blank 
# convert categorical data 
# survived_counts = data.Survived.value_counts()
# print(survived_counts)
# pclass_values = data.Pclass.unique()
# print(pclass_values)
# sex_values = data.Sex.unique()
# print(sex_values)
# age_values = data.Age.unique()
# print(age_values)
# sibsbp_values = data.SibSp.unique()
# print(sibsbp_values)
# parch_values = data.Parch.unique()
# print(parch_values)
# ticket_values = data.Ticket.unique()
# print(ticket_values)
# fare_values = data.Fare.unique()
# print(fare_values)
# cabin_values = data.Cabin.unique()
# print(cabin_values)
# embarked_values = data.Embarked.unique()
# print(embarked_values)



