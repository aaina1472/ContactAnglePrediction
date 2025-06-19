import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor

# ----------------------
# Load and prepare data
# ----------------------
st.set_page_config(page_title="Contact Angle Predictor", layout="centered")

df = pd.read_csv("download.csv")
df = df.dropna().reset_index(drop=True)

# Features and target
numeric_features = ["Texture diameter", "Texture length", "Texture depth", "Texture pitch", "Roughness factor"]
X = df[numeric_features]
y = df["Contact angle"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), numeric_features)]
)

# Model pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(random_state=42))
])

model_pipeline.fit(X, y)

# ----------------------
# Streamlit UI
# ----------------------
st.title("🔬 Contact Angle Predictor")
st.markdown("Enter texture parameters below to predict the **contact angle** using a trained Gradient Boosting model.")

# Input UI layout
col1, col2 = st.columns(1)

with col1:
    diameter = st.number_input("🔵 Texture Diameter (µm)", value=25.0)
    depth = st.number_input("🔵 Texture Depth (µm)", value=5.0)
    roughness = st.number_input("🔵 Roughness Factor", min_value=1.0, max_value=2.0, value=1.5, step=0.01)

with col2:
    length = st.number_input("🔵 Texture Length (µm)", value=25.0)
    pitch = st.number_input("🔵 Texture Pitch (µm)", value=25.0)

# Prediction
st.markdown("---")
if st.button("📐 Predict Contact Angle"):
    input_df = pd.DataFrame([{
        "Texture diameter": diameter,
        "Texture length": length,
        "Texture depth": depth,
        "Texture pitch": pitch,
        "Roughness factor": roughness
    }])
    
    prediction = model_pipeline.predict(input_df)[0]
    st.success(f"🧪 Predicted Contact Angle: **{prediction:.2f}°**")

