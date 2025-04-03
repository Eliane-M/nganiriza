import io
from fastapi import Body, FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
import os
import time
import pandas as pd
import joblib
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from pydantic import BaseModel
import logging
from datetime import datetime
from src.api.database import client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("api.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Import pipeline functions
from src.preprocessing import preprocess_prediction_data, preprocess_training_data, ALL_FEATURES
from src.model import define_model, train_model, load_model
from src.prediction import make_predictions, predict_from_raw_data, interpret_prediction


# Initialize FastAPI app
app = FastAPI(
    title="Machine Learning Model API",
    description="API for prediction, data upload, and model retraining",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
db = client["TeenPregnancyDB"]
collection = db["uploaded_data"]

logger = logging.getLogger(__name__)

# Path constants
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions")
VISUALIZATIONS_DIR = os.path.join(MODEL_DIR, "visualizations")

# Ensure directories exist
for dir_path in [MODEL_DIR, DATA_DIR, TRAIN_DIR, TEST_DIR, PREDICTIONS_DIR, VISUALIZATIONS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# File paths
MODEL_PATH = os.path.join(MODEL_DIR, "nn_model_5.h5")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
MAPPING_PATH = os.path.join(MODEL_DIR, "category_mapping.pkl")
X_TRAIN_PATH = os.path.join(TRAIN_DIR, "X_train.pkl")
Y_TRAIN_PATH = os.path.join(TRAIN_DIR, "y_train.pkl")
X_TEST_PATH = os.path.join(TEST_DIR, "X_test.pkl")
Y_TEST_PATH = os.path.join(TEST_DIR, "y_test.pkl")
X_VAL_PATH = os.path.join(TRAIN_DIR, "X_val.pkl")
Y_VAL_PATH = os.path.join(TRAIN_DIR, "y_val.pkl")
X_NEW_PATH = os.path.join(TRAIN_DIR, "X_new.pkl")
Y_NEW_PATH = os.path.join(TRAIN_DIR, "y_new.pkl")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
FEATURE_IMPORTANCE_PATH = os.path.join(VISUALIZATIONS_DIR, "feature_importance.png")
TRAINING_HISTORY_PATH = os.path.join(VISUALIZATIONS_DIR, "training_history.png")
MODEL_METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
PERFORMANCE_LOG_PATH = os.path.join(MODEL_DIR, "performance_log.json")

# Load preprocessor at startup
try:
    if os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        logger.info("Preprocessor loaded successfully")
    else:
        preprocessor = None
        logger.warning("Preprocessor not found at path: %s", PREPROCESSOR_PATH)
except Exception as e:
    logger.error("Error loading preprocessor: %s", str(e))
    preprocessor = None

# State variable for tracking retraining
retraining_in_progress = False

# Serve static files
app.mount("/static", StaticFiles(directory=VISUALIZATIONS_DIR), name="static")

# Pydantic models
class PredictionInput(BaseModel):
    features: dict

class PredictionBatchInput(BaseModel):
    batch_features: List[dict]

class TrainingParams(BaseModel):
    epochs: int = 10
    batch_size: int = 32
    validation_split: float = 0.2

# Helper functions
def log_request_metrics(start_time, endpoint, status_code):
    duration = time.time() - start_time
    if os.path.exists(PERFORMANCE_LOG_PATH):
        with open(PERFORMANCE_LOG_PATH, 'r') as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = {"requests": []}
    else:
        log_data = {"requests": []}
    log_data["requests"].append({
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
        "duration_seconds": duration,
        "status_code": status_code
    })
    durations = [req["duration_seconds"] for req in log_data["requests"]]
    log_data["stats"] = {
        "total_requests": len(durations),
        "avg_duration": sum(durations) / len(durations) if durations else 0,
        "min_duration": min(durations) if durations else 0,
        "max_duration": max(durations) if durations else 0,
        "last_updated": datetime.now().isoformat()
    }
    with open(PERFORMANCE_LOG_PATH, 'w') as f:
        json.dump(log_data, f, indent=2)
    return duration

def generate_feature_importance(model, feature_names=None):
    try:
        weights = model.layers[0].get_weights()[0]
        importance = np.mean(np.abs(weights), axis=1)
        if feature_names is None:
            feature_names = ALL_FEATURES
        feat_importance = pd.DataFrame({
            'Feature': feature_names[:len(importance)],
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feat_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(FEATURE_IMPORTANCE_PATH)
        plt.close()
        return feat_importance.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error generating feature importance: {str(e)}")
        return None

def generate_training_history_plot(history):
    try:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'])
            plt.legend(['Train', 'Validation'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'])
            plt.legend(['Train', 'Validation'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.tight_layout()
        plt.savefig(TRAINING_HISTORY_PATH)
        plt.close()
        return True
    except Exception as e:
        logger.error(f"Error generating training history plot: {str(e)}")
        return False

async def retrain_model_task(epochs=10, batch_size=32, validation_split=0.2):
    global preprocessor, retraining_in_progress
    retraining_in_progress = True
    try:
        logger.info("Starting model retraining")
        if not (os.path.exists(X_TRAIN_PATH) and os.path.exists(Y_TRAIN_PATH)):
            logger.error("Initial training data not found")
            retraining_in_progress = False
            return False
        
        # Load initial training data
        with open(X_TRAIN_PATH, "rb") as f:
            X_train_df = pickle.load(f)
        with open(Y_TRAIN_PATH, "rb") as f:
            y_train = pickle.load(f)
        
        # Append uploaded data if available
        if os.path.exists(X_NEW_PATH):
            with open(X_NEW_PATH, "rb") as f:
                X_new_df = pickle.load(f)
            X_combined_df = pd.concat([X_train_df, X_new_df], ignore_index=True)
            logger.info(f"Combined {len(X_train_df)} original samples with {len(X_new_df)} new samples")
        else:
            X_combined_df = X_train_df
            logger.info(f"Using only original {len(X_train_df)} samples")
        
        if os.path.exists(Y_NEW_PATH):
            with open(Y_NEW_PATH, "rb") as f:
                y_new = pickle.load(f)
            y_combined = np.concatenate([y_train, y_new]) if y_train.shape[1] == y_new.shape[1] else y_train
        else:
            y_combined = y_train
        
        # Preprocess training data
        X_processed, y_combined = preprocess_training_data(
            X_combined_df.assign(Risk_Category=y_combined),
            preprocessor_path=PREPROCESSOR_PATH,
            mapping_path=MAPPING_PATH
        )
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        
        # Train the model
        model_params = {
            "optimization": "adam",
            "regularization_type": "l1_l2",
            "regularization_strength": 0.01,
            "early_stopping": True,
            "learning_rate": 0.001
        }
        model, history = train_model(
            X_processed, y_combined,
            X_val=X_VAL_PATH, y_val=Y_VAL_PATH,
            model_path=MODEL_PATH,
            epochs=epochs,
            batch_size=batch_size,
            **model_params
        )
        
        # Generate visualizations
        generate_training_history_plot(history)
        generate_feature_importance(model)
        
        # Evaluate on test data if available
        if os.path.exists(X_TEST_PATH) and os.path.exists(Y_TEST_PATH):
            with open(X_TEST_PATH, "rb") as f:
                X_test_df = pickle.load(f)
            with open(Y_TEST_PATH, "rb") as f:
                y_test = pickle.load(f)
            X_test_processed = preprocess_prediction_data(X_test_df, PREPROCESSOR_PATH)
            metrics = evaluate_model(model, X_test_processed, y_test, CONFUSION_MATRIX_PATH)
            with open(MODEL_METRICS_PATH, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Model evaluation completed. Accuracy: {metrics.get('accuracy', 'N/A')}")
        
        logger.info("Model retraining completed successfully")
        return True
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        return False
    finally:
        retraining_in_progress = False

def evaluate_model(model, X_test, y_test, confusion_matrix_path):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    probabilities = model.predict(X_test)
    predicted_classes = np.argmax(probabilities, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_test_classes, predicted_classes)),
        "precision": float(precision_score(y_test_classes, predicted_classes, average='weighted')),
        "recall": float(recall_score(y_test_classes, predicted_classes, average='weighted')),
        "f1_score": float(f1_score(y_test_classes, predicted_classes, average='weighted'))
    }
    cm = confusion_matrix(y_test_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(confusion_matrix_path)
    plt.close()
    return metrics

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Machine Learning Model API is running", "version": "1.0.0"}

@app.post("/predict")
async def predict(data: PredictionInput):
    print("Received data:", data.dict())
    start_time = time.time()
    try:
        if not preprocessor:
            raise HTTPException(status_code=500, detail="Preprocessor not initialized. Please retrain first.")
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=404, detail="Model not found. Please train a model first.")
        
        probabilities, predicted_classes = predict_from_raw_data(
            data.features, PREPROCESSOR_PATH, MODEL_PATH,
            save_path=os.path.join(PREDICTIONS_DIR, "prediction.json")
        )
        result = interpret_prediction(probabilities[0])
        
        duration = log_request_metrics(start_time, "predict", 200)
        result["processing_time"] = duration
        return result
    except Exception as e:
        status_code = 500
        log_request_metrics(start_time, "predict", status_code)
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=status_code, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(data: PredictionBatchInput):
    start_time = time.time()
    try:
        if not preprocessor:
            raise HTTPException(status_code=500, detail="Preprocessor not initialized. Please retrain first.")
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=404, detail="Model not found. Please train a model first.")
        
        df = pd.DataFrame(data.batch_features)
        probabilities, predicted_classes = predict_from_raw_data(
            df, PREPROCESSOR_PATH, MODEL_PATH,
            save_path=os.path.join(PREDICTIONS_DIR, "batch_prediction.json")
        )
        result = {
            "predictions": [interpret_prediction(prob) for prob in probabilities]
        }
        
        duration = log_request_metrics(start_time, "predict_batch", 200)
        result["processing_time"] = duration
        result["samples_per_second"] = len(data.batch_features) / duration if duration > 0 else 0
        return result
    except Exception as e:
        status_code = 500
        log_request_metrics(start_time, "predict_batch", status_code)
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=status_code, detail=f"Batch prediction error: {str(e)}")

@app.post("/upload")
async def upload(
    x_file: UploadFile = File(...),
    y_file: Optional[UploadFile] = File(None),
    replace_existing: bool = Form(False)
):
    start_time = time.time()
    try:
        # Validate file types
        if not (x_file.filename.endswith(".pkl") or x_file.filename.endswith(".csv")):
            raise HTTPException(status_code=400, detail="X file must be a .pkl or .csv file")
        
        if y_file and not (y_file.filename.endswith(".pkl") or y_file.filename.endswith(".csv")):
            raise HTTPException(status_code=400, detail="Y file must be a .pkl or .csv file")

        # Process CSV file (Save to MongoDB)
        if x_file.filename.endswith(".csv"):
            x_contents = await x_file.read()
            df = pd.read_csv(io.StringIO(x_contents.decode("utf-8")))
            data = df.to_dict(orient="records")  # Convert CSV to list of dictionaries
            
            if replace_existing:
                collection.delete_many({})  # Clear existing data if replace_existing is True
            
            collection.insert_many(data)  # Insert new records
            logger.info(f"CSV data inserted into MongoDB: {len(data)} records")
        
        # Process Y file if provided
        if y_file and y_file.filename.endswith(".csv"):
            y_contents = await y_file.read()
            df_y = pd.read_csv(io.StringIO(y_contents.decode("utf-8")))
            data_y = df_y.to_dict(orient="records")
            
            if replace_existing:
                collection.delete_many({})  # Optionally clear existing data
            
            collection.insert_many(data_y)  # Insert new records
            logger.info(f"Y CSV data inserted into MongoDB: {len(data_y)} records")

        result = {
            "message": "Files uploaded and stored successfully",
            "x_file": x_file.filename,
            "y_file": y_file.filename if y_file else None,
            "replace_existing": replace_existing
        }

        duration = time.time() - start_time
        result["processing_time"] = duration
        return result

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks, params: TrainingParams = Body(None)):
    global retraining_in_progress
    start_time = time.time()
    try:
        if retraining_in_progress:
            raise HTTPException(status_code=409, detail="Retraining already in progress")
        
        if params is None:
            params = TrainingParams()
        
        background_tasks.add_task(
            retrain_model_task,
            epochs=params.epochs,
            batch_size=params.batch_size,
            validation_split=params.validation_split
        )
        
        result = {
            "message": "Model retraining started in the background",
            "training_params": params.dict(),
            'processing_time': time.time() - start_time        
        }

        return JSONResponse(status_code=202, content=result)
    except Exception as e:
        status_code = 500 if not isinstance(e, HTTPException) else e.status_code
        log_request_metrics(start_time, "retrain", status_code)
        logger.error(f"Retraining initiation error: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=status_code, detail=f"Retraining initiation error: {str(e)}")

# Remaining endpoints (unchanged from your original)
@app.get("/retrain/status")
async def retrain_status():
    try:
        result = {"retraining_in_progress": retraining_in_progress}
        if not retraining_in_progress and os.path.exists(MODEL_METRICS_PATH):
            with open(MODEL_METRICS_PATH, 'r') as f:
                metrics = json.load(f)
            result["metrics"] = metrics
        return result
    except Exception as e:
        logger.error(f"Error fetching retraining status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching retraining status: {str(e)}")

@app.get("/visualizations")
async def get_visualizations():
    try:
        visualizations = {
            "confusion_matrix": os.path.exists(CONFUSION_MATRIX_PATH),
            "feature_importance": os.path.exists(FEATURE_IMPORTANCE_PATH),
            "training_history": os.path.exists(TRAINING_HISTORY_PATH)
        }
        for key in visualizations:
            if visualizations[key]:
                visualizations[f"{key}_url"] = f"/visualize/{key}"
        return visualizations
    except Exception as e:
        logger.error(f"Error fetching visualizations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching visualizations: {str(e)}")

@app.get("/visualize/{viz_type}")
async def visualize(viz_type: str):
    viz_map = {
        "confusion_matrix": CONFUSION_MATRIX_PATH,
        "feature_importance": FEATURE_IMPORTANCE_PATH,
        "training_history": TRAINING_HISTORY_PATH
    }
    if viz_type not in viz_map:
        raise HTTPException(status_code=404, detail=f"Visualization '{viz_type}' not found")
    viz_path = viz_map[viz_type]
    if not os.path.exists(viz_path):
        raise HTTPException(status_code=404, detail=f"Visualization '{viz_type}' file not found")
    return FileResponse(viz_path, media_type="image/png")

@app.get("/metrics")
async def get_metrics():
    try:
        result = {}
        if os.path.exists(MODEL_METRICS_PATH):
            with open(MODEL_METRICS_PATH, 'r') as f:
                result["model_metrics"] = json.load(f)
        if os.path.exists(PERFORMANCE_LOG_PATH):
            with open(PERFORMANCE_LOG_PATH, 'r') as f:
                perf_data = json.load(f)
                result["performance"] = perf_data.get("stats", {})
        if not result:
            raise HTTPException(status_code=404, detail="No metrics available")
        return result
    except Exception as e:
        logger.error(f"Error fetching metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching metrics: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        model_ready = os.path.exists(MODEL_PATH)
        preprocessor_ready = preprocessor is not None
        status = {
            "status": "healthy" if model_ready and preprocessor_ready else "degraded",
            "model_ready": model_ready,
            "preprocessor_ready": preprocessor_ready,
            "retraining_in_progress": retraining_in_progress,
            "timestamp": datetime.now().isoformat()
        }
        import platform
        status["system_info"] = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "tensorflow_version": tf.__version__
        }
        return status
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)