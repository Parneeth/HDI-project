import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(csv_path):
    """
    Loads HDI dataset, handles any missing values, splits data 
    into features and target, and performs a 75/25 train-test split.
    """
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Check for missing values and fill using median
    for col in ["Life_Expectancy", "Expected_Schooling", "Mean_Schooling", "GNI_Per_Capita"]:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
            
    # Features and target
    feature_cols = ["Life_Expectancy", "Expected_Schooling", "Mean_Schooling", "GNI_Per_Capita"]
    X = df[feature_cols]
    y = df["HDI"]
    
    # Split into 75% train and 25% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols
