# ----------- Imports -----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ----------- Load Dataset -----------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv"
df = pd.read_csv(url, encoding="latin1")

print("\n--- Dataset Head ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# ----------- Column Cleaning -----------
df.columns = [
    c.split("(")[0]
     .strip()
     .replace(" ", "_")
     .lower()
    for c in df.columns
]

df = df.rename(columns={
    "rented_bike_count": "count",
    "date": "datetime"
})

# ----------- Datetime Handling -----------
df["datetime"] = pd.to_datetime(df["datetime"], format="%d/%m/%Y")
df = df.sort_values("datetime")

# ----------- Feature Engineering -----------
df["month"] = df["datetime"].dt.month
df["day_of_week"] = df["datetime"].dt.day_of_week
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

df.drop("datetime", axis=1, inplace=True)

# ----------- EDA -----------
numerical_eda_features = [
    'temperature', 'humidity', 'wind_speed',
    'visibility', 'dew_point_temperature',
    'solar_radiation', 'rainfall', 'snowfall'
]

plt.figure(figsize=(20, 15))
for i, feature in enumerate(numerical_eda_features):
    plt.subplot(3, 3, i + 1)
    sns.scatterplot(x=feature, y="count", data=df, alpha=0.4)
    plt.title(f"Count vs {feature}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(
    df[numerical_eda_features + ["count"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.show()

# ----------- Train Test Split -----------
X = df.drop("count", axis=1)
y = df["count"]

numerical_features = [
    'hour', 'temperature', 'humidity', 'wind_speed',
    'visibility', 'dew_point_temperature',
    'solar_radiation', 'rainfall', 'snowfall',
    'month', 'day_of_week'
]

categorical_features = [
    "seasons", "holiday",
    "functioning_day", "is_weekend"
]

# ----------- Preprocessing Pipelines -----------
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1190
)

# ==============================
# Linear Regression Model
# ==============================
linear_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
r2_linear = r2_score(y_test, y_pred_linear)

print("\n--- Linear Regression ---")
print(f"RMSE: {rmse_linear:.2f}")
print(f"R2  : {r2_linear:.4f}")

plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred_linear, alpha=0.3)
plt.plot([0, 3000], [0, 3000], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Linear Regression Predictions")
plt.grid(True)
plt.show()

# Random Forest Model
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=30,
        random_state=1190,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\n--- Random Forest Regressor ---")
print(f"RMSE: {rmse_rf:.2f}")
print(f"R2  : {r2_rf:.4f}")

plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred_rf, alpha=0.3)
plt.plot([0, 3000], [0, 3000], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Random Forest Predictions")
plt.grid(True)
plt.show()
