from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd

from .utils import load_data, split_data, load_model

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache data and model
df = load_data()
X_train, X_test, y_train, y_test = split_data(df)
model = load_model()

# Define response endpoints
@app.get("/data")
def get_test_data():
    return {
        "X_test": X_test.to_dict(orient="records"),
        "y_test": y_test.to_dict(orient="records")
    }

@app.get("/feature_names")
def get_feature_names():
    return list(X_test.columns)

@app.get("/predict_all")
def predict_all():
    predictions = model.predict(X_test)
    return pd.DataFrame(predictions, columns=["Heating_Load", "Cooling_Load"]).to_dict(orient="records")
