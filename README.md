# Breast Cancer Classification

## Overview
This project implements a **machine learning model served through a FastAPI web API**.  
It predicts whether a breast cancer tumor is **benign (0)** or **malignant (1)** based on diagnostic features.  

This is a modified version of the original lab template:
- Dataset changed from **Iris** → **Breast Cancer Wisconsin dataset**
- Model changed from **DecisionTreeClassifier** → **RandomForestClassifier**
- Added endpoints: `/metrics` and `/health`
- API returns both **prediction** and **confidence probability**


## Dataset
- Source: `sklearn.datasets.load_breast_cancer`
- Samples: 569
- Features: 30 numeric tumor characteristics (mean radius, mean texture, perimeter, area, smoothness, etc.)
- Target:
  - `0` → Benign (non-cancerous)
  - `1` → Malignant (cancerous)


## Project Structure
```
mlops_labs
└── fastapi_lab1
    ├── assets/
    ├── fastapi_lab1_env/
    ├── model/
    │   └── cancer_model.pkl
    |   |__metrics.pkl
    ├── src/
    │   ├── __init__.py
    │   ├── data.py
    │   ├── main.py
    │   ├── predict.py
    │   └── train.py
    ├── README.md
    └── requirements.txt
    

```
## Setup & Installation

1. Clone the repo or download the folder.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv fastapi_lab1_env
   source fastapi_lab1_env/bin/activate   # Mac/Linux
   fastapi_lab1_env\Scripts\activate      # Windows
3. Install dependencies:
   ```
   pip install -r requirements.txt
5. Train the model (saves cancer_model.pkl + metrics.pkl in model/):
   ```
   python -m src.train
6. Run the FastAPI server:
   ```
   uvicorn src.main:app --reload
7. Open Swagger UI for testing:
   ```
   http://127.0.0.1:8000/docs

## API Endpoints
#### Root

`GET /`

Returns welcome message.

Example:

`{"message": "Breast Cancer Classification API is running"}`

#### Single Prediction

`POST /predict`

Input: JSON with a list of 30 numeric features.

Example request:
```
{
  "features": [14.1,20.0,92.0,600.0,0.09,0.10,0.07,0.05,0.18,0.06,
               0.3,1.0,2.5,25.0,0.005,0.02,0.02,0.01,0.01,0.003,
               15.0,25.0,100.0,800.0,0.12,0.2,0.2,0.1,0.3,0.07]
}
```

Response:
`

{"prediction": 1, "probability": 0.95}
`

#### Batch Prediction

`POST /predict_batch`

Input: List of JSON objects.

Example request:
```
[
  {"features":[14.1,20.0,92.0,600.0,0.09,0.10,0.07,0.05,0.18,0.06,
               0.3,1.0,2.5,25.0,0.005,0.02,0.02,0.01,0.01,0.003,
               15.0,25.0,100.0,800.0,0.12,0.2,0.2,0.1,0.3,0.07]},
  {"features":[12.3,15.6,80.0,500.0,0.08,0.09,0.06,0.04,0.15,0.05,
               0.25,0.8,2.1,20.0,0.004,0.018,0.019,0.012,0.008,0.002,
               14.0,22.0,90.0,700.0,0.11,0.18,0.19,0.09,0.25,0.06]}
]

```
Response:
`
{
  "predictions": [
    {"prediction": 1, "probability": 0.95},
    {"prediction": 0, "probability": 0.92}
  ]
}
`
#### Health Check

`GET /health`

Returns API and model info.

Example:
`
{"status": "ok", "model": "RandomForestClassifier", "version": "1.0"}
`
#### Metrics

`GET /metrics`

Returns model evaluation results.

Example:

`{"accuracy": 0.95, "f1_score": 0.94}`

