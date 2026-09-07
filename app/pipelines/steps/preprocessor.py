"""
Data preprocessing and feature engineering
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import logging

logger = logging.getLogger(__name__)


def preprocess_data(train_df, val_df, test_df, sampling_strategy="smote", target_ratio=0.1):
    """Preprocess data with feature engineering and sampling"""
    target_col = "Class"
    
    # Feature engineering
    def engineer_features(df):
        df = df.copy()
        if "Time" in df.columns:
            df["Hour"] = (df["Time"] % 86400) // 3600
            df = df.drop(columns=["Time"])
        if "Amount" in df.columns:
            df["Amount_log"] = np.log1p(df["Amount"])
        return df
    
    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)
    
    # Split features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE if requested
    if sampling_strategy == "smote":
        smote = SMOTE(sampling_strategy=target_ratio, random_state=42)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        logger.info(f"Applied SMOTE. New training size: {len(X_train_scaled)}")
    
    feature_names = X_train.columns.tolist()
    
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, feature_names, scaler
