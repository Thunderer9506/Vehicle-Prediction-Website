import pandas as pd
import numpy as np
import re
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- Load and preprocess data ---
data = pd.read_csv('cleanedData.csv')
data.drop(['exterior_color', 'interior_color'], axis=1, inplace=True)

# --- Feature engineering ---
data['engine_liters'] = data['engine'].str.extract(r'(\d\.?\d*)L', flags=re.IGNORECASE).astype(float)
data['is_turbo'] = data['engine'].str.contains('Turbo', case=False, na=False).astype(int)
data['is_hybrid'] = data['engine'].str.contains('Hybrid', case=False, na=False).astype(int)
data['valve_count'] = data['engine'].str.extract(r'(\d{1,2})V', flags=re.IGNORECASE).astype(float)
data['engine_liters'].fillna(data['engine_liters'].median(), inplace=True)
data['valve_count'].fillna(data['valve_count'].median(), inplace=True)

data['gears'] = data['transmission'].str.extract(r'(\d{1,2})-Speed', flags=re.IGNORECASE).astype(float)
conditions = [
    data['transmission'].str.contains('CVT', case=False, na=False),
    data['transmission'].str.contains('Automatic', case=False, na=False)
]
data['transmission_type'] = np.select(conditions, ['CVT', 'Automatic'], default='Other')
data['gears'].fillna(data['gears'].median(), inplace=True)

model_counts = data['model'].value_counts()
trim_counts = data['trim'].value_counts()
rare_models = model_counts[model_counts < 10].index
rare_trims = trim_counts[trim_counts < 5].index
data['model_cleaned'] = data['model'].replace(rare_models, 'Other')
data['trim_cleaned'] = data['trim'].replace(rare_trims, 'Other')

current_year = 2025
data['age'] = current_year - data['year']

data.drop(['engine', 'transmission', 'model', 'trim'], axis=1, inplace=True)

# --- Label Encoding ---
categorical_cols = [
    'make', 'fuel', 'body', 'drivetrain', 
    'model_cleaned', 'trim_cleaned', 'transmission_type'
]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop('price', axis=1)
y = data['price']
feature_columns = X.columns.tolist()

# --- Scale ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train model ---
model = RandomForestRegressor(
    n_estimators=300, max_features=0.57, random_state=42, 
    oob_score=True, max_depth=20
)
model.fit(X_scaled, y)

# --- Evaluate ---
y_pred = model.predict(X_scaled)
metrics = {
    "R² Score": r2_score(y, y_pred),
    "MAE": mean_absolute_error(y, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
    "OOB Score": model.oob_score_
}

# --- Save artifacts ---
artifacts = {
    "model": model,
    "scaler": scaler,
    "label_encoders": label_encoders,
    "feature_columns": feature_columns,
    "rare_models": rare_models,
    "rare_trims": rare_trims,
    "current_year": current_year,
    "metrics": metrics
}

with open('artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("✅ Model and artifacts saved successfully!")
