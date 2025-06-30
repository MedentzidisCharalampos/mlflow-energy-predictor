import openml
import pandas as pd
import mlflow.pyfunc
from sklearn.model_selection import train_test_split

def load_data():
    dataset = openml.datasets.get_dataset(1472)
    df, *_ = dataset.get_data()
    df.rename(columns={"y1": "Heating_Load", "y2": "Cooling_Load"}, inplace=True)
    df['Heating_Load'] = pd.to_numeric(df['Heating_Load'], errors='coerce')
    df['Cooling_Load'] = pd.to_numeric(df['Cooling_Load'], errors='coerce')
    df.dropna(inplace=True)
    return df

def split_data(df):
    X = df.drop(["Heating_Load", "Cooling_Load"], axis=1)
    y = df[["Heating_Load", "Cooling_Load"]]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def load_model():
    return mlflow.pyfunc.load_model("models:/energy_model/Production")