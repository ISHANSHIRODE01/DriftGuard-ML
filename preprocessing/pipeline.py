"""
Student Performance - Preprocessing Module
==========================================
Single Source of Truth for raw data cleaning and transformation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import joblib
from pathlib import Path

# --- Centralized Cleaning Logic ---
def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies mandated raw data transformations.
    
    Rules:
    - Convert binary 'yes'/'no' columns to 1/0 integers.
    """
    df_clean = df.copy()
    
    binary_features = [
        'schoolsup', 'famsup', 'paid', 'activities', 
        'nursery', 'higher', 'internet', 'romantic'
    ]
    
    for col in binary_features:
        if col in df_clean.columns:
            # Handle string 'yes'/'no'
            mask_yes = df_clean[col].astype(str).str.lower() == 'yes'
            df_clean[col] = mask_yes.astype(int)
            
    return df_clean

class StudentPerformancePreprocessor:
    """
    Comprehensive preprocessing pipeline for student performance prediction.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.preprocessor = None
        self.feature_names = None
        self._define_feature_groups()
        
    def _define_feature_groups(self):
        # Ordinal features
        self.ordinal_features = {
            'Medu': [0, 1, 2, 3, 4],
            'Fedu': [0, 1, 2, 3, 4],
            'traveltime': [1, 2, 3, 4],
            'studytime': [1, 2, 3, 4],
            'famrel': [1, 2, 3, 4, 5],
            'freetime': [1, 2, 3, 4, 5],
            'goout': [1, 2, 3, 4, 5],
            'Dalc': [1, 2, 3, 4, 5],
            'Walc': [1, 2, 3, 4, 5],
            'health': [1, 2, 3, 4, 5]
        }
        
        # Binary features (yes/no)
        self.binary_features = [
            'schoolsup', 'famsup', 'paid', 'activities', 
            'nursery', 'higher', 'internet', 'romantic'
        ]
        
        # Nominal features
        self.nominal_features = [
            'school', 'sex', 'address', 'famsize', 'Pstatus',
            'Mjob', 'Fjob', 'reason', 'guardian'
        ]
        
        # Numerical features
        self.numerical_features = [
            'age', 'failures', 'absences', 'G1', 'G2'
        ]
        
        self.target = 'G3'
        
    def create_preprocessing_pipeline(self):
        ordinal_transformer = OrdinalEncoder(
            categories=[self.ordinal_features[col] for col in self.ordinal_features.keys()],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        
        onehot_transformer = OneHotEncoder(
            drop='first',
            sparse_output=False,
            handle_unknown='ignore'
        )
        
        numerical_transformer = StandardScaler()
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('ordinal', ordinal_transformer, list(self.ordinal_features.keys())),
                ('onehot', onehot_transformer, self.nominal_features),
                ('numerical', numerical_transformer, self.numerical_features)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )
        return self.preprocessor
    
    def prepare_data(self, df):
        """Prepare data using Centralized Logic."""
        # 1. Apply centralized cleaning
        df_clean = clean_raw_data(df)
        
        # 2. Separate features and target
        if self.target in df_clean.columns:
            X = df_clean.drop(columns=[self.target])
            y = df_clean[self.target]
            return X, y
        else:
            return df_clean, None

    def create_stratified_bins(self, y, n_bins=5):
        return pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
    
    def split_data(self, X, y, test_size=0.2, val_size=0.125):
        y_bins = self.create_stratified_bins(y)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y_bins, random_state=self.random_state
        )
        y_temp_bins = self.create_stratified_bins(y_temp)
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, stratify=y_temp_bins, random_state=self.random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def fit_transform_pipeline(self, X_train, X_val, X_test):
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_val_transformed = self.preprocessor.transform(X_val)
        X_test_transformed = self.preprocessor.transform(X_test)
        self.feature_names = self._get_feature_names()
        return X_train_transformed, X_val_transformed, X_test_transformed
    
    def _get_feature_names(self):
        feature_names = []
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'ordinal':
                feature_names.extend(columns)
            elif name == 'onehot':
                if hasattr(transformer, 'get_feature_names_out'):
                    feature_names.extend(transformer.get_feature_names_out(columns))
                else:
                    feature_names.extend(columns)
            elif name == 'numerical':
                feature_names.extend(columns)
            elif name == 'remainder':
                feature_names.extend(self.binary_features)
        return feature_names
    
    def save_pipeline(self, filepath='models/preprocessor.pkl'):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, filepath)
    
    def load_pipeline(self, filepath='models/preprocessor.pkl'):
        self.preprocessor = joblib.load(filepath)
        self.feature_names = self._get_feature_names()
