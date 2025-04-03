import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import pickle
from pymongo import MongoClient
import gridfs
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from urllib.parse import quote_plus
from src.preprocessing import preprocess_prediction_data, preprocess_training_data, ALL_FEATURES, create_preprocessor
from src.model import define_model, train_model
from src.prediction import predict_from_raw_data

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount('/static', StaticFiles(directory='./models/static'), name='static')

load_dotenv()

UPLOAD_FOLDER = './models/uploads'
STATIC_FOLDER = './models/static'
MODEL_PATH = './models/nn_model.h5'
X_TRAIN_PATH = './data/train/X_train.pkl'
Y_TRAIN_PATH = './data/train/y_train.pkl'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(X_TRAIN_PATH), exist_ok=True)

# MongoDB configuration
ATLAS_USER = os.getenv("ATLAS_USER")
ATLAS_PASSWORD = os.getenv("ATLAS_PASSWORD")
ATLAS_DB = os.getenv("ATLAS_DB")
ATLAS_CLUSTER = os.getenv("ATLAS_CLUSTER")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

encoded_user = quote_plus(ATLAS_USER)
encoded_password = quote_plus(ATLAS_PASSWORD)
MONGODB_URI = f"mongodb+srv://{encoded_user}:{encoded_password}@{ATLAS_CLUSTER}.mongodb.net/{ATLAS_DB}?retryWrites=true&w=majority"

# Connect to MongoDB Atlas
try:
    client = MongoClient(MONGODB_URI)
    db = client[ATLAS_DB]
    fs = gridfs.GridFS(db)
    print("Connected to MongoDB Atlas successfully!")
except Exception as e:
    print(f"Failed to connect to MongoDB Atlas: {str(e)}")
    raise

# Pydantic model for prediction input
class PredictionInput(BaseModel):
    Age: int
    Education_Level: str
    Ubudehe_Category: str
    Family_Income_Rwf: float
    Healthcare_Access_Score: int
    Sexual_Education_Hours: float
    Contraceptive_Use: str
    Peer_Influence: int
    Parental_Involvement: int
    Community_Resources: int

# Global preprocessor (initialized during retrain)
preprocessor = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(data: PredictionInput):
    global preprocessor
    
    logger.info("=== /predict Endpoint Called ===")
    logger.info("Incoming Data: %s", data.dict())
    
    if preprocessor is None:
        logger.error("Preprocessor not initialized")
        raise HTTPException(status_code=500, detail="Preprocessor not initialized. Please retrain the model first.")
    
    df = pd.DataFrame([data.dict()])
    logger.info("DataFrame: %s", df.to_dict(orient='records'))
    
    X_processed = preprocessor.transform(df)
    logger.info("Preprocessed Data: %s", X_processed)
    
    probabilities, predicted_classes = predict_from_raw_data(df, preprocessor, MODEL_PATH)
    logger.info("Probabilities: %s", probabilities)
    logger.info("Predicted Classes: %s", predicted_classes)
    
    class_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    predicted_label = class_mapping.get(predicted_classes[0], "Unknown Risk")
    
    response = {
        "Risk_Category": predicted_label,
        "Probabilities": probabilities[0].tolist()
    }
    logger.info("Response: %s", response)
    logger.info("=== /predict Endpoint Finished ===")
    
    return response


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Save to MongoDB and clean up
        file_id = fs.put(open(file_path, 'rb'), filename=file.filename)
        os.remove(file_path)  # Cleanup after upload
        print(f"File {file.filename} uploaded to MongoDB with ID: {file_id}")
        return {"message": f"File {file.filename} uploaded successfully", "file_id": str(file_id)}
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)  # Ensure cleanup on error
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/retrain")
async def retrain():
    global preprocessor
    
    # Find the most recently uploaded file
    latest_file = db.fs.files.find_one(sort=[("uploadDate", -1)])  # Sort by uploadDate, descending
    if not latest_file:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Retrieve the latest file from GridFS
    file_data = fs.get(latest_file['_id']).read()
    temp_path = os.path.join(UPLOAD_FOLDER, latest_file['filename'])
    try:
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(file_data)
        df = pd.read_csv(temp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)  # Cleanup temporary file

    # Use the single DataFrame for retraining
    combined_df = df
    
    # Create and fit preprocessor
    preprocessor = create_preprocessor()
    X_scaled = preprocessor.fit_transform(combined_df[ALL_FEATURES])
    y = combined_df['Risk_Category']
    y_encoded = pd.get_dummies(y).to_numpy()  # One-hot encode target
    
    # Save raw data for visualizations
    with open(X_TRAIN_PATH, 'wb') as f:
        pickle.dump(combined_df[ALL_FEATURES], f)
    with open(Y_TRAIN_PATH, 'wb') as f:
        pickle.dump(combined_df['Risk_Category'], f)
    
    model, _ = train_model(X_scaled, y_encoded, model_path=MODEL_PATH)
    
    return {"message": f"Model retrained successfully using {latest_file['filename']}"}

@app.get("/visualizations/{plot_type}")
async def get_visualizations(plot_type: str):
    # Load training data from pickle files
    try:
        with open(X_TRAIN_PATH, 'rb') as f:
            X_train = pickle.load(f)
        with open(Y_TRAIN_PATH, 'rb') as f:
            y_train = pickle.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Training data files not found")

    # Reconstruct DataFrame
    df = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train, columns=ALL_FEATURES)
    df['Risk_Category'] = y_train

    plots = []
    
    if plot_type == "education":
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='Education_Level', hue='Risk_Category')
        plt.xticks(rotation=45)
        plt.title('Education Level Distribution')
        plot_path1 = f"{STATIC_FOLDER}/{plot_type}_count.png"
        plt.savefig(plot_path1)
        plt.close()
        plots.append(plot_path1)

        plt.figure(figsize=(8, 6))
        education_risk = df.groupby(['Education_Level', 'Risk_Category']).size().unstack().fillna(0)
        education_risk.plot(kind='bar', stacked=True)
        plt.title('Education Level vs Risk Category')
        plt.xticks(rotation=45)
        plot_path2 = f"{STATIC_FOLDER}/{plot_type}_bar.png"
        plt.savefig(plot_path2)
        plt.close()
        plots.append(plot_path2)

    elif plot_type == "contraceptive":
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='Contraceptive_Use', hue='Risk_Category')
        plt.xticks(rotation=45)
        plt.title('Contraceptive Use Distribution')
        plot_path1 = f"{STATIC_FOLDER}/{plot_type}_count.png"
        plt.savefig(plot_path1)
        plt.close()
        plots.append(plot_path1)

        plt.figure(figsize=(8, 6))
        contraceptive_risk = df.groupby(['Contraceptive_Use', 'Risk_Category']).size().unstack().fillna(0)
        contraceptive_risk.plot(kind='bar', stacked=True)
        plt.title('Contraceptive Use vs Risk Category')
        plt.xticks(rotation=45)
        plot_path2 = f"{STATIC_FOLDER}/{plot_type}_bar.png"
        plt.savefig(plot_path2)
        plt.close()
        plots.append(plot_path2)

    elif plot_type == "income":
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x='Risk_Category', y='Family_Income_Rwf')
        plt.title('Family Income vs Risk Category')
        plot_path1 = f"{STATIC_FOLDER}/{plot_type}_box.png"
        plt.savefig(plot_path1)
        plt.close()
        plots.append(plot_path1)

        plt.figure(figsize=(8, 6))
        sns.violinplot(data=df, x='Risk_Category', y='Family_Income_Rwf')
        plt.title('Family Income Distribution by Risk Category')
        plot_path2 = f"{STATIC_FOLDER}/{plot_type}_violin.png"
        plt.savefig(plot_path2)
        plt.close()
        plots.append(plot_path2)

    else:
        raise HTTPException(status_code=404, detail="Invalid plot type")
    
    # Cleanup static folder (keep max 10 files)
    files = sorted(os.listdir(STATIC_FOLDER), key=lambda x: os.path.getctime(os.path.join(STATIC_FOLDER, x)))
    while len(files) > 10:
        os.remove(os.path.join(STATIC_FOLDER, files.pop(0)))

    return JSONResponse(content={
        "plot1": f"{BACKEND_URL}/static/{os.path.basename(plots[0])}",
        "plot2": f"{BACKEND_URL}/static/{os.path.basename(plots[1])}"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)