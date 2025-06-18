import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor

# Define features
numeric_features = ["Texture diameter", "Texture length", "Texture depth", "Texture pitch", "Roughness factor"]
categorical_features = ["Texture shape"]

# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Load your dataset
df = pd.read_csv("download.csv")
df = df.dropna().reset_index(drop=True)

# Split into features and target
X = df.drop(columns=["Contact angle"])
y = df["Contact angle"]

# Define and train the model
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(random_state=42))
])
model_pipeline.fit(X, y)

# Streamlit UI
st.title("Contact Angle Prediction App 🧪")

st.write("Enter the following texture parameters to predict the contact angle:")

# Input form
diameter = st.slider("Texture Diameter", 10.0, 50.0, 25.0)
length = st.slider("Texture Length", 10.0, 50.0, 25.0)
depth = st.slider("Texture Depth", 1.0, 10.0, 5.0)
pitch = st.slider("Texture Pitch", 10.0, 50.0, 25.0)
roughness = st.slider("Roughness Factor", 1.0, 2.0, 1.5)
shape = st.selectbox("Texture Shape", df["Texture shape"].dropna().unique())

# Predict button
if st.button("Predict Contact Angle"):
    input_df = pd.DataFrame([{
        "Texture diameter": diameter,
        "Texture length": length,
        "Texture depth": depth,
        "Texture pitch": pitch,
        "Roughness factor": roughness,
        "Texture shape": shape
    }])
    
    prediction = model_pipeline.predict(input_df)[0]
    st.success(f"Predicted Contact Angle: {prediction:.2f}°")
