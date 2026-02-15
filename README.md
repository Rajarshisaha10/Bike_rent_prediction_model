🚲 Seoul Bike Demand Prediction – End-to-End ML Pipeline

A complete machine learning pipeline for predicting hourly bike rental demand using weather and seasonal data from the UCI Seoul Bike Dataset.

This project demonstrates:

Real-world dataset handling

Feature engineering

EDA with visual insights

Clean preprocessing pipelines

Model comparison (Linear vs Random Forest)

Proper evaluation metrics

📊 Dataset

Source:
UCI Machine Learning Repository – Seoul Bike Sharing Demand Dataset

URL used in project:

https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv

The dataset contains:

Hourly bike rental counts

Weather conditions

Seasonal and holiday indicators

8760 rows (1 year of hourly data)

🧠 Problem Statement

Predict the number of rented bikes (count) based on:

Weather conditions

Time-based features

Seasonal patterns

Holiday information

This is a supervised regression problem.

⚙️ Project Workflow
1️⃣ Data Loading & Cleaning

Loaded dataset from UCI URL

Cleaned column names

Renamed:

Rented Bike Count → count

Date → datetime

Converted datetime to pandas format

2️⃣ Feature Engineering

Created:

month

day_of_week

is_weekend

Dropped raw datetime after extraction.

3️⃣ Exploratory Data Analysis (EDA)

✔ Scatter plots of rental count vs weather variables
✔ Correlation heatmap

Key observations:

Temperature strongly correlates with bike rentals

Rainfall and snowfall negatively impact demand

Seasonal trends are visible

4️⃣ Train-Test Split
test_size = 0.2
random_state = 1190

Ensures reproducibility.

5️⃣ Preprocessing Pipeline

Used ColumnTransformer + Pipeline for clean production-style preprocessing.

Numerical Features:

Median Imputation

Standard Scaling

Categorical Features:

Most Frequent Imputation

One-Hot Encoding (handle_unknown="ignore")

This ensures:

No data leakage

Clean separation of concerns

Production-ready structure

🤖 Models Implemented
🔹 1. Linear Regression

Baseline model

Assumes linear relationship

Fast and interpretable

Metrics:

RMSE

R² Score

Prediction vs Actual visualization included.

🔹 2. Random Forest Regressor

30 estimators

Parallel processing (n_jobs=-1)

Captures non-linear patterns

Typically outperforms linear regression

Metrics:

RMSE

R² Score

📈 Evaluation Metrics
RMSE (Root Mean Squared Error)

Measures average prediction error magnitude.

R² Score

Explains variance captured by the model.

🧪 Example Output
--- Linear Regression ---
RMSE: XXXX.XX
R2  : 0.XX

--- Random Forest Regressor ---
RMSE: XXXX.XX
R2  : 0.XX

(Random Forest generally performs better due to non-linear modeling.)

🛠 Tech Stack

Python

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

📂 Project Structure
seoul-bike-regression/
│
├── bike_regression.py
├── README.md
🔥 Key Strengths of This Project

✔ Proper ML pipeline structure
✔ Real-world dataset
✔ Feature engineering
✔ Model comparison
✔ Clean visualization
✔ Reproducibility
✔ Industry-standard preprocessing

This is not a notebook experiment.
This is structured ML engineering.

🚀 Possible Improvements

Hyperparameter tuning (GridSearchCV)

Cross-validation

Feature importance visualization

SHAP explainability

XGBoost / LightGBM comparison

Time-series aware splitting

Model saving with joblib

Deployment via FastAPI

👤 Author

Rajarshi Saha
B.Tech – VIT Chennai
