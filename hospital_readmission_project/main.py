# Hospital Readmission Prediction - Assignment Code
# Author: [Your Name]

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report

# --- 1. Data Simulation (Replacing real data for this assignment) ---
def generate_dummy_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 90, n_samples),
        'num_lab_procedures': np.random.randint(1, 100, n_samples),
        'num_medications': np.random.randint(1, 40, n_samples),
        'time_in_hospital': np.random.randint(1, 14, n_samples),
        'has_insurance': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
        # Target: 0 = No Readmission, 1 = Readmission within 30 days
        'readmitted': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]) 
    }
    return pd.DataFrame(data)

# --- 2. Preprocessing ---
def preprocess_data(df):
    # Separate features and target
    X = df.drop('readmitted', axis=1)
    y = df['readmitted']
    
    # Split Data: 70% Train, 15% Val, 15% Test
    # First split into Train and Temp
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    # Split Temp into Validation and Test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Scaling (Normalization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

# --- 3. Model Development ---
def train_model(X_train, y_train):
    # Using Random Forest as justified in the report
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model

# --- 4. Evaluation ---
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    print("--- Model Evaluation ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Specific Metrics
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")

# --- Execution Flow ---
if __name__ == "__main__":
    print("Starting AI Development Workflow...")
    
    # 1. Load Data
    df = generate_dummy_data()
    print(f"Data Loaded. Shape: {df.shape}")
    
    # 2. Preprocess
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df)
    
    # 3. Train
    model = train_model(X_train, y_train)
    print("Model Training Complete.")
    
    # 4. Evaluate
    evaluate_model(model, X_test, y_test)