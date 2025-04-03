from typing import Optional, Tuple
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import pickle
import json
from .preprocessing import preprocess_prediction_data, ALL_FEATURES

def make_predictions(model_path: str, X_processed: np.ndarray, save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using a trained model
    
    Args:
        model_path: Path to the saved model
        X_processed: Preprocessed input features
        save_path: Path to save predictions (optional)
        
    Returns:
        probabilities: Prediction probabilities for each class
        predicted_classes: Predicted class indices
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not isinstance(X_processed, np.ndarray):
            raise ValueError("X_processed must be a numpy array")
        
        loaded_model = tf.keras.models.load_model(model_path)
        probabilities = loaded_model.predict(X_processed)
        predicted_classes = np.argmax(probabilities, axis=1)
        
        if save_path:
            results = {
                "probabilities": probabilities.tolist(),
                "predicted_classes": predicted_classes.tolist()
            }
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(results, f)
            print(f"Predictions saved to {save_path}")
        
        return probabilities, predicted_classes
    
    except FileNotFoundError as fe:
        print(f"FileNotFoundError in make_predictions: {fe}")
        raise
    except ValueError as ve:
        print(f"ValueError in make_predictions: {ve}")
        raise
    except Exception as e:
        print(f"Error in make_predictions: {e}")
        raise

def predict_from_raw_data(data, preprocessor=None, model_path: str = None, save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess raw data and make predictions
    
    Args:
        data: Raw data as a dictionary or pandas DataFrame
        preprocessor: Preprocessor object (optional, if None, assumes data is preprocessed)
        model_path: Path to the saved model
        save_path: Path to save predictions (optional)
        
    Returns:
        probabilities: Prediction probabilities for each class
        predicted_classes: Predicted class indices
    """
    try:
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError("Data must be a dictionary or pandas DataFrame")
        
        missing_cols = set(ALL_FEATURES) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        if preprocessor is not None:
            X_processed = preprocessor.transform(df)
        else:
            X_processed = df.to_numpy()  # Assume data is already preprocessed if no preprocessor
        
        return make_predictions(model_path, X_processed, save_path)
    
    except Exception as e:
        print(f"Error in predict_from_raw_data: {e}")
        raise

def interpret_prediction(probability, class_mapping=None):
    """
    Interpret model prediction probabilities
    
    Args:
        probability: Array of class probabilities
        class_mapping: Dictionary mapping class indices to names
        
    Returns:
        result: Dictionary with prediction details
    """
    if class_mapping is None:
        class_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    
    predicted_class = np.argmax(probability)
    
    result = {
        "predicted_class": predicted_class,
        "predicted_label": class_mapping[predicted_class],
        "confidence": float(probability[predicted_class]),
        "probabilities": {
            class_mapping[i]: float(prob) 
            for i, prob in enumerate(probability)
        }
    }
    
    return result