# Machine Learning Model API

This project demonstrates the deployment of a machine learning model through a FastAPI-based REST API. The application allows users to make predictions, upload new data, trigger model retraining, and visualize model performance.

## Features

- **Model prediction**: Make predictions on single data points or batches  
- **Data upload**: Upload new training data to enhance the model  
- **Model retraining**: Trigger retraining with custom parameters  
- **Visualizations**: View model performance through various visualizations  
- **Containerization**: Docker support for easy deployment  

## Project Structure

```
nganiriza/
│
├── README.md
│
├── notebook/
│    └── Nganiriza.ipynb
│
├── src/
│    ├── api/
│    ├── preprocessing.py
│    ├── model.py
│    └── prediction.py
│
├── data/
│    ├── train/
│    └── test/
│
├── models/
│    ├── nn_model_5.h5
│    ├── static/
│    └── uploads/
│
├── app.py                  # Contains the endpoints
│
└── nganiriza_frontend      # The react app for the frontend
```


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

NB: The .env file that didn't get pushed contains these
ATLAS_USER=emunezero
ATLAS_PASSWORD=qxSTjxixi1SbVmLf
ATLAS_DB=TeenPregnancyDB
ATLAS_CLUSTER=summative.bjbmaxp
BACKEND_URL=http://localhost:8000


### The deployed links

The frontend is live at https://nganiriza-web-app.onrender.com/ with the [deployed backend](https://nganiriza.onrender.com)

The endpoints created are the following:
  - /predict
  - /upload
  - /retrain
  - visualizations

This is a link to the [Demo Video](https://youtu.be/h_D39599LxU)


You can sue this dataset for testing 
[teenage_pregnancies_in_Rwanda_data.csv](https://github.com/user-attachments/files/19645689/teenage_pregnancies_in_Rwanda_data.csv)
