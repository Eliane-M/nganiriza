import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import pickle

def define_model(input_dim, optimization='adam', regularization_type='l1_l2', 
                 regularization_strength=0.01, early_stopping=True, learning_rate=0.001, verbose=False):
    """
    Define a neural network classification model
    
    Args:
        input_dim: Input dimension (number of features after preprocessing)
        optimization: Optimizer to use
        regularization_type: Type of regularization ('l1', 'l2', 'l1_l2', or None)
        regularization_strength: Strength of regularization
        early_stopping: Whether to use early stopping
        learning_rate: Learning rate for optimizer
        
    Returns:
        model: Compiled Keras model
        callbacks: List of callbacks
    """
    # Initialize the model
    model = Sequential()

    def get_regularizer(reg_type, reg_strength):
        if reg_type == 'l1':
            return l1(reg_strength) if reg_strength > 0 else None
        elif reg_type == 'l2':
            return l2(reg_strength) if reg_strength > 0 else None
        elif reg_type == 'l1_l2':
            return l1_l2(l1=reg_strength, l2=reg_strength) if reg_strength > 0 else None
        else:
            return None

    # First dense layer with optional regularization
    model.add(Dense(32, activation='relu', input_shape=(input_dim,),
                    kernel_regularizer=get_regularizer(regularization_type, regularization_strength)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Second dense layer
    model.add(Dense(16, activation='relu',
                    kernel_regularizer=get_regularizer(regularization_type, regularization_strength)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Third dense layer
    model.add(Dense(8, activation='relu',
                    kernel_regularizer=get_regularizer(regularization_type, regularization_strength)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Output layer with 3 neurons for multi-class classification
    model.add(Dense(3, activation='softmax'))

    # Compile the model    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) if optimization == 'adam' else optimization
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Print model summary if verbose is True
    if verbose:
        print(model.summary())

    # Define callbacks for early stopping if required
    callbacks = []
    if early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))

    return model, callbacks

def train_model(X_train, y_train, X_val=None, y_val=None, model_path='../models/nn_model_5.h5', 
                epochs=100, batch_size=32, **model_params):
    """
    Train and save the model
    
    Args:
        X_train: Training features
        y_train: Training targets (should be one-hot encoded)
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        model_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_params: Parameters to pass to define_model
        
    Returns:
        model: Trained Keras model
        history: Training history
    """
    try:
        input_dim = X_train.shape[1]
        model, callbacks = define_model(input_dim, **model_params)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            if X_val.shape[1] != X_train.shape[1] or y_val.shape[1] != y_train.shape[1]:
                raise ValueError("Validation data dimensions do not match training data")
            validation_data = (X_val, y_val)
        
        # Train the model
        history = model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        return model, history
    
    except Exception as e:
        print(f"Error in train_model: {e}")
        raise

def load_model(model_path):
    """
    Load a saved model
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        model: Loaded Keras model
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model = tf.keras.models.load_model(model_path)
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise