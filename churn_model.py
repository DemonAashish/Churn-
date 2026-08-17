import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder # Fixed capitalization
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

print("hello")

# Generate synthetic dataset (removing the unused pd.read_csv line)
x_dummy, y_dummy = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=15, 
    n_redundant=5, 
    random_state=40
)

# Create DataFrame
df = pd.DataFrame(x_dummy, columns=[f'feature_{i}' for i in range(20)]) 
df['Churn'] = y_dummy

print(f'Dataset loaded with {df.shape[0]} records and {df.shape[1]} columns.')

# X and y are the features and target variable respectively
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    # Note: use_label_encoder=False is deprecated in newer XGBoost versions but works for older ones
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42) 
}

# Model training and evaluation
best_model_name = ''  
best_model_accuracy = 0

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'\n--- {model_name} ---')
    print(f'Accuracy: {accuracy:.4f}')
    print(classification_report(y_test, y_pred))
    
    if accuracy > best_model_accuracy:
        best_model_accuracy = accuracy
        best_model_name = model_name

print(f'\nBest Model: {best_model_name} with Accuracy: {best_model_accuracy:.4f}')