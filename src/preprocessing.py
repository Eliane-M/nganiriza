import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
from typing import Tuple, Optional, Dict

# Define feature columns
NUMERIC_COLUMNS = [
    'Age', 'Family_Income_Rwf', 'Healthcare_Access_Score',
    'Sexual_Education_Hours', 'Peer_Influence', 'Parental_Involvement', 'Community_Resources'
]

CATEGORICAL_COLUMNS = ['Education_Level', 'Ubudehe_Category', 'Contraceptive_Use']

ALL_FEATURES = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
TARGET_COLUMN = 'Risk_Category'

def create_category_mapping(df: pd.DataFrame, categorical_columns: list) -> Dict[str, str]:
    category_mapping = {}
    for column in categorical_columns:
        unique_values = df[column].unique()
        for value in unique_values:
            one_hot_column_name = f"{column}_{value}"
            category_mapping[f"{column}_{value}"] = one_hot_column_name
    return category_mapping

def create_preprocessor() -> Pipeline:
    numeric_transformer = SimpleImputer(strategy='mean')
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_COLUMNS),
            ('cat', categorical_transformer, CATEGORICAL_COLUMNS)
        ]
    )
    
    preprocessor = Pipeline(steps=[
        ('column_transformer', column_transformer),
        ('scaler', StandardScaler())
    ])
    
    return preprocessor

def preprocess_training_data(df: pd.DataFrame, preprocessor_path: str, mapping_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if df.empty:
        raise ValueError("Training data is empty.")
    
    required_columns = set(ALL_FEATURES + [TARGET_COLUMN])
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        raise ValueError(f"Missing columns: {missing_cols}")
    
    category_mapping = create_category_mapping(df, CATEGORICAL_COLUMNS)
    with open(mapping_path, 'wb') as file:
        pickle.dump(category_mapping, file)
    
    df = df.dropna(subset=[TARGET_COLUMN])
    if df.empty:
        raise ValueError("No training data after dropping rows with missing targets.")
    
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]
    
    preprocessor = create_preprocessor()
    X_processed = preprocessor.fit_transform(X)
    
    with open(preprocessor_path, 'wb') as file:
        pickle.dump(preprocessor, file)
    
    return X_processed, y.to_numpy()

def preprocess_prediction_data(df: pd.DataFrame, preprocessor_path: str) -> Optional[np.ndarray]:
    if df.empty:
        raise ValueError("Prediction data is empty.")
    
    required_columns = set(ALL_FEATURES)
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        raise ValueError(f"Missing columns: {missing_cols}")
    
    with open(preprocessor_path, 'rb') as file:
        preprocessor = pickle.load(file)
    
    X_processed = preprocessor.transform(df)
    
    return X_processed
