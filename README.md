# Machine Learning Model API

This project demonstrates the deployment of a machine learning model through a FastAPI-based REST API. The application allows users to make predictions, upload new data, trigger model retraining, and visualize model performance.

## Features

- **Model prediction**: Make predictions on single data points or batches  
- **Data upload**: Upload new training data to enhance the model  
- **Model retraining**: Trigger retraining with custom parameters  
- **Visualizations**: View model performance through various visualizations  
- **Performance monitoring**: Track API performance metrics  
- **Load testing**: Simulate high traffic with Locust  
- **Containerization**: Docker support for easy deployment  

## Project Structure

ml-model-api/ │ ├── README.md # Project documentation ├── app.py # FastAPI application entry point ├── Dockerfile # Docker container definition ├── docker-compose.yml # Docker Compose configuration ├── requirements.txt # Python dependencies │ ├── src/ # Source code │ ├── preprocessing.py # Data preprocessing functions │ ├── model.py # Model definition and training │ ├── prediction.py # Prediction functions │ ├── data/ # Data storage │ ├── train/ # Training data │ ├── test/ # Test data │ ├── predictions/ # Stored predictions │ ├── models/ # Model storage │ ├── visualizations/ # Model visualizations │ └── locust/ # Load testing └── locustfile.py # Locust test configuration



## Setup and Installation

### Prerequisites

- Python 3.10  
- Docker and Docker Compose  
- Git  

### Local Development Setup

1. Clone the repository:

   ```
   git clone https://github.com/Eliane-M/nganiriza.git
   ```
   ```
   cd nganiriza
   ```

2. Create a virtual environment and install dependencies:

   ```
   python -m venv venv
   ```
   ```
   source venv/bin/activate    #On Windows: venv/Scripts/activate
   ```
   ```
   pip install -r requirements.txt
   ```


3. Run the application

   ```
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

### The deployed links

The frontend is live at https://nganiriza-web-app.onrender.com/ with the [deployed backend](https://nganiriza.onrender.com)

The endpoints created are the following:
  - /predict
  - /upload
  - /retrain
  - visualizations

This is a link to the [Demo Video]()
   
