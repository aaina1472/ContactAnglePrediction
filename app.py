import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor

# ----------------------
# Load and prepare data
# ----------------------
st.set_page_config(page_title="Contact Angle Predictor", layout="centered")

df = pd.read_csv("download.csv")
df = df.dropna().reset_index(drop=True)

# Separate numeric and categorical features
numeric_features = ["Texture diameter", "Texture length", "Texture depth", "Texture pitch", "Roughness factor"]
categorical_features = ["Texture shape"]

X = df[numeric_features + categorical_features]
y = df["Contact angle"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
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
st.markdown("## 🔧 Input Texture Parameters")

# Input fields
diameter = st.number_input("🔵 Texture Diameter (µm)", value=25.0)
length = st.number_input("🔵 Texture Length (µm)", value=25.0)
depth = st.number_input("🔵 Texture Depth (µm)", value=5.0)
pitch = st.number_input("🔵 Texture Pitch (µm)", value=25.0)
roughness = st.number_input("🔵 Roughness Factor", min_value=1.0, max_value=2.0, value=1.5, step=0.01)

# New input: Texture Shape (Dropdown)
shape_options = df["Texture shape"].unique().tolist()
shape = st.selectbox("🧩 Texture Shape", options=shape_options)


# Predict button
st.markdown("---")
if st.button("📐 Predict Contact Angle"):
    input_df = pd.DataFrame([{
        "Texture diameter": diameter,
        "Texture length": length,
        "Texture depth": depth,
        "Texture pitch": pitch,
        "Roughness factor": roughness,
        "Texture shape": shape
    }])

    prediction = model_pipeline.predict(input_df)[0]
    st.success(f"🧪 Predicted Contact Angle: **{prediction:.2f}°**")

