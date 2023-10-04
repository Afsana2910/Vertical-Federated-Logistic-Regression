import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Load the dataset
file_path = "C:/Users/P70077043/Documents/Contribution_in_VFL/new_experiment/heart_disease.csv"
df = pd.read_csv(file_path)

X = df.drop('output', axis=1)
y = df['output']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Train the Logistic Regression model
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f1_score(y_test, y_pred))
print(f"Accuracy: {accuracy*100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))
