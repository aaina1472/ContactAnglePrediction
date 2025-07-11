# -*- coding: utf-8 -*-
"""NewContactAngle1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Bqyd6_gxsxYp7OgOp3-Jx3MiPjsZWGVz

# Importing Libraries
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

"""# Importing Dataset"""

import pandas as pd
df = pd.read_csv('download.csv')
df

df.info()

df.shape

"""# Cleaning Data"""

df = df.dropna().reset_index(drop=True)
df

df.shape

df['Texture shape'].unique()

"""# Feature Selection"""

# Define features and target
x = df.drop(columns=["Contact angle"])  #rest all features except target feature
y = df["Contact angle"]

x

y

# Define numeric and categorical columns
numeric_features = ["Texture diameter", "Texture length", "Texture depth", "Texture pitch", "Roughness factor"]
categorical_features = ["Texture shape"]

"""# Finding Correlations"""

# Define numeric columns
numeric_columns = ["Texture diameter", "Texture length", "Texture depth", "Texture pitch", "Roughness factor", "Contact angle"]

corr_matrix = df[numeric_columns].corr()
corr_matrix['Contact angle'].sort_values(ascending=False)

import matplotlib.pyplot as plt
import seaborn as sns # Ensure seaborn is imported for heatmap

plt.figure(figsize=(8, 6))
# Select only numeric columns for correlation calculation
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

"""# Creating Column Transformer"""

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

"""# Creating Pipeline"""

# Import different models
pipelines = {
    "Linear Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]),
    "KNN Regressor": Pipeline([
        ("preprocessor", preprocessor),
        ("model", KNeighborsRegressor(n_neighbors=5))
    ]),
    "Gradient Boosting": Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
}

pipeline.fit(x, y)

for name, pipe in pipelines.items():
    print(f"🧩 {name}")
    display(pipe)

for name, pipeline in pipelines.items():
    scores = cross_val_score(pipeline, x, y, cv=5, scoring="r2")
    print(f"{name} -> Mean R²: {scores.mean():.4f}, Std Dev: {scores.std():.4f}")

from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np

# Custom scorers for MSE and RMSE (note sklearn doesn't have built-in RMSE scorer)
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)  # negative MSE for cross_val_score
rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

for name, pipeline in pipelines.items():
    r2_scores = cross_val_score(pipeline, x, y, cv=5, scoring="r2")
    neg_mse_scores = cross_val_score(pipeline, x, y, cv=5, scoring="neg_mean_squared_error")
    # RMSE can be derived from neg_mse_scores
    rmse_scores = np.sqrt(-neg_mse_scores)

    print(f"{name} -> Mean R²: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    print(f"{name} -> Mean MSE: {-neg_mse_scores.mean():.4f} ± {neg_mse_scores.std():.4f}")
    print(f"{name} -> Mean RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
    print()

"""# Train-Test Split using Linear Regression"""

from sklearn.model_selection import train_test_split

# Pehle X (features) aur y (target) define karo
X = df.drop(columns=["Contact angle"])
y = df["Contact angle"]

# Ab train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
lr=LinearRegression()
lr.fit(X_train_processed,y_train)
y_lr_train_pred=lr.predict(X_train_processed)
y_lr_test_pred=lr.predict(X_test_processed)

from sklearn.metrics import mean_squared_error, r2_score
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

print('LR MSE (Train): ', lr_train_mse)
print('LR R2 (Train): ', lr_train_r2)
print('LR MSE (Test): ', lr_test_mse)
print('LR R2 (Test): ', lr_test_r2)

"""# Train-Test Split using Gradient Boosting"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pick a model pipeline
model_name = "Gradient Boosting"
model_pipeline = pipelines[model_name]

# Fit the model
model_pipeline.fit(X_train, y_train)

# Predict
y_pred = model_pipeline.predict(X_test)

# Compare actual vs predicted
comparison_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
print(comparison_df.head(10))  # Show first 10 rows

import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set_style('darkgrid')

# Get and fit each pipeline before predicting
pipeline_lr = pipelines["Linear Regression"].fit(X_train, y_train)
pipeline_knn = pipelines["KNN Regressor"].fit(X_train, y_train)
pipeline_gbr = pipelines["Gradient Boosting"].fit(X_train, y_train)
pipeline_rf = pipelines["Random Forest"].fit(X_train, y_train)

# Predict on test data using the fitted pipelines
y_lr_pred = pipeline_lr.predict(X_test)
y_knn_pred = pipeline_knn.predict(X_test)
y_gbr_pred = pipeline_gbr.predict(X_test)
y_rf_pred = pipeline_rf.predict(X_test)

# Plot all in one figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Linear Regression
axs[0, 0].scatter(y_test, y_lr_pred, color='blue', alpha=0.6)
axs[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axs[0, 0].set_title('Linear Regression')
axs[0, 0].set_xlabel('Actual')
axs[0, 0].set_ylabel('Predicted')

# KNN Regressor
axs[0, 1].scatter(y_test, y_knn_pred, color='orange', alpha=0.6)
axs[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axs[0, 1].set_title('KNN Regressor')
axs[0, 1].set_xlabel('Actual')
axs[0, 1].set_ylabel('Predicted')

# Gradient Boosting
axs[1, 0].scatter(y_test, y_gbr_pred, color='green', alpha=0.6)
axs[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axs[1, 0].set_title('Gradient Boosting')
axs[1, 0].set_xlabel('Actual')
axs[1, 0].set_ylabel('Predicted')

# Random Forest
axs[1, 1].scatter(y_test, y_rf_pred, color='red', alpha=0.6)
axs[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axs[1, 1].set_title('Random Forest')
axs[1, 1].set_xlabel('Actual')
axs[1, 1].set_ylabel('Predicted')

plt.tight_layout()
plt.suptitle("Scatter Plots: Actual vs Predicted (Test Data)", fontsize=16, y=1.03)
plt.show()

# prompt: predict with random values

# Generate random values for prediction (using the same feature names)

import random

known_categories = ['Parallel Dimple', 'Staggered dimple', 'Micro grid', 'Groove'] # Example known categories
random_data = {
    "Texture diameter": [np.random.uniform(10, 50)],  # Example range
    "Texture length": [np.random.uniform(10, 50)],    # Example range
    "Texture depth": [np.random.uniform(1, 10)],      # Example range
    "Texture pitch": [np.random.uniform(10, 50)],     # Example range
    "Roughness factor": [np.random.uniform(1, 2)],   # Example range
    "Texture shape": [random.choice(known_categories)] # Example choices
}

random_df = pd.DataFrame(random_data)

# Use the trained model pipeline to predict on the random data
random_prediction = model_pipeline.predict(random_df)

print(f"\nPrediction for random data: {random_prediction[0]:.4f}")