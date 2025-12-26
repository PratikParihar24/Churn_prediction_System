import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_and_prepare_data(file_path):
    """
    Complete preprocessing pipeline for Telco Churn Data.
    """
    # 1. Load Data
    df = pd.read_csv(file_path)
    
    # 2. Fix the TotalCharges 'Trap'
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    
    # 3. Drop useless columns (IDs don't help prediction)
    df.drop('customerID', axis=1, inplace=True)
    
    # 4. Binary Encoding (Map Yes/No to 1/0)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Map Gender separately
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    
    # 5. One-Hot Encoding for remaining categorical columns
    # This handles 'Contract', 'PaymentMethod', 'InternetService', etc.
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_final = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    
    # 6. Define Features (X) and Target (y)
    X = df_final.drop('Churn', axis=1)
    y = df_final['Churn']
    
    # 7. Split Data
    # Stratify=y is CRITICAL here to keep the 26% churn ratio in both sets!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 8. Scale Numerical Features
    # We only scale tenure and charges so they have a mean of 0 and std of 1.
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    # Test the script
    X_train, X_test, y_train, y_test, _ = clean_and_prepare_data('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print(f"Data Prepared. Training Shape: {X_train.shape}")