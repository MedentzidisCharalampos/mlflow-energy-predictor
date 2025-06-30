# 🔋 Energy Efficiency Prediction System
This project provides a production-ready architecture for predicting the Heating Load and Cooling Load of buildings using a pre-trained XGBoost model. It includes:

✅ A FastAPI backend serving predictions and dataset info

✅ A Streamlit dashboard that visualizes predictions vs ground truth

✅ A registered MLflow model loaded from the Model Registry

✅ Clean modular code and RESTful design

## 🧩 Project Structure

├── backend/  
│   ├── main.py              # FastAPI app  
│   ├── utils.py             # Data loading, model loading, split functions  
│   ├── train.py             # Model training and MLflow logging  
├── frontend/  
│   ├── app.py               # Streamlit dashboard (calls FastAPI)    
├── requirements.txt         # Dependencies  
└── README.md  

## 📦 Install Requirements

### Create and activate virtual environment
```python 
conda create -n energy-predict python=3.10 -y
conda activate energy-predict
```

### Install dependencies
```python 
 pip install -r requirements.txt
```
## 🔁 Train and Register the Model (optional if already registered)

```
python train.py
```


Loads data from OpenML and Trains a MultiOutput XGBoost model. Logs it to MLflow and registers under the name energy_model

## 🚀 Run the FastAPI Backend

```
cd backend
uvicorn main:app --reload
```
Runs at: http://localhost:8000

Docs at: http://localhost:8000/docs

Endpoints:

GET /data – Returns X_test, y_test

GET /predict_all – Returns predictions on X_test

GET /feature_names – Lists feature names

## 🎯 Launch the Streamlit Frontend
```
cd frontend
streamlit run app.py
```

Loads test data and predictions via API and Allows interactive selection of samples and features.

Displays:

 - Heating Load vs Ground Truth

- Cooling Load vs Ground Truth

- Feature explorer

📊 Dataset
Source: OpenML Dataset 1472

Features:

- Relative Compactness

- Surface Area

- Wall Area

- Roof Area

- Overall Height

- Orientation

- Glazing Area

- Glazing Area Distribution

Targets:

- Heating_Load

- Cooling_Load

## ✅ Highlights  
🧠 Model	MultiOutput XGBoost for regression  
📦 Backend	FastAPI with batch predictions and CORS  
🌐 Frontend	Streamlit dashboard with Altair charts  
🔄 MLflow	Used to register and serve trained model  
📊 Dataset Source	OpenML (ID: 1472)  

📦 Example API Request

GET http://localhost:8000/predict_all
```
[
  {
    "Heating_Load": 13.42,
    "Cooling_Load": 19.85
  },
  ...
]
```

## 📄 License
MIT License – use freely, cite respectfully.

