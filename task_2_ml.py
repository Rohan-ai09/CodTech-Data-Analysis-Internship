# Task 2: Predictive Analysis using Machine Learning (Titanic Dataset)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("titanic.csv")

# Preprocessing
data["Age"].fillna(data["Age"].median(), inplace=True)
data["Embarked"].fillna("S", inplace=True)
data.drop("Cabin", axis=1, inplace=True)
data = pd.get_dummies(data, columns=["Sex", "Embarked", "Pclass"], drop_first=True)
data.drop(["Name", "Ticket", "PassengerId"], axis=1, inplace=True)

# Feature and target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluation
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
