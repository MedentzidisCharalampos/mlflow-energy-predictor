import streamlit as st
import pandas as pd
import altair as alt
import requests
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Energy Prediction Dashboard", layout="wide")
st.title("🔋 Energy Efficiency Prediction Dashboard")
st.markdown("This dashboard compares model predictions with ground truth values and allows feature exploration.")

# --------- Load test data and predictions from FastAPI backend ---------

@st.cache_data
def get_data():
    response = requests.get(f"{API_URL}/data").json()
    return pd.DataFrame(response["X_test"]), pd.DataFrame(response["y_test"])

@st.cache_data
def get_predictions():
    response = requests.get(f"{API_URL}/predict_all").json()
    return pd.DataFrame(response)

X_test, y_test = get_data()
preds = get_predictions()

# --------- User selection ---------

selected_indices = st.multiselect(
    "🎯 Select test sample indices to visualize",
    options=list(range(len(X_test))),
    default=list(range(min(20, len(X_test))))
)

# --------- Visualizations & Metrics ---------

if selected_indices:
    selected_gt = y_test.iloc[selected_indices].reset_index(drop=True)
    selected_preds = preds.iloc[selected_indices].reset_index(drop=True)
    selected_samples = X_test.iloc[selected_indices].reset_index(drop=True)
    sample_index = list(range(len(selected_gt)))

    # --- Metrics ---
    rmse_heating = np.sqrt(mean_squared_error(selected_gt["Heating_Load"], selected_preds["Heating_Load"]))
    r2_heating = r2_score(selected_gt["Heating_Load"], selected_preds["Heating_Load"])

    rmse_cooling = np.sqrt(mean_squared_error(selected_gt["Cooling_Load"], selected_preds["Cooling_Load"]))
    r2_cooling = r2_score(selected_gt["Cooling_Load"], selected_preds["Cooling_Load"])

    st.markdown("### 📈 Evaluation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔥 Heating RMSE", f"{rmse_heating:.2f}")
    col2.metric("🔥 Heating R²", f"{r2_heating:.2f}")
    col3.metric("❄️ Cooling RMSE", f"{rmse_cooling:.2f}")
    col4.metric("❄️ Cooling R²", f"{r2_cooling:.2f}")

    # --- Heating Chart ---
    st.subheader("🔥 Heating Load: Prediction vs Ground Truth")
    heating_df = pd.DataFrame({
        "Index": sample_index,
        "Ground Truth": selected_gt["Heating_Load"],
        "Prediction": selected_preds["Heating_Load"]
    }).melt(id_vars="Index", var_name="Type", value_name="Value")

    heating_chart = alt.Chart(heating_df).mark_line(point=True).encode(
        x="Index:O",
        y="Value:Q",
        color="Type:N"
    ).properties(title="Heating Load - Prediction vs Ground Truth")

    st.altair_chart(heating_chart, use_container_width=True)

    # --- Cooling Chart ---
    st.subheader("❄️ Cooling Load: Prediction vs Ground Truth")
    cooling_df = pd.DataFrame({
        "Index": sample_index,
        "Ground Truth": selected_gt["Cooling_Load"],
        "Prediction": selected_preds["Cooling_Load"]
    }).melt(id_vars="Index", var_name="Type", value_name="Value")

    cooling_chart = alt.Chart(cooling_df).mark_line(point=True).encode(
        x="Index:O",
        y="Value:Q",
        color="Type:N"
    ).properties(title="Cooling Load - Prediction vs Ground Truth")

    st.altair_chart(cooling_chart, use_container_width=True)

    # --- Feature Explorer ---
    st.subheader("📈 Feature Value Explorer")
    feature_list = list(X_test.columns)
    selected_feature = st.selectbox("🔍 Choose a feature to visualize", feature_list)

    feature_df = pd.DataFrame({
        "Index": sample_index,
        selected_feature: selected_samples[selected_feature]
    })

    feature_chart = alt.Chart(feature_df).mark_bar().encode(
        x="Index:O",
        y=alt.Y(f"{selected_feature}:Q", title=selected_feature),
        color=alt.value("#1f77b4")
    ).properties(title=f"{selected_feature} - Values for Selected Samples")

    st.altair_chart(feature_chart, use_container_width=True)
