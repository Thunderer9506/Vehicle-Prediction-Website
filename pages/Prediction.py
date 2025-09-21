# main.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- Page Configuration (should be the first st command) ---
st.set_page_config(
    page_title="Vehicle Price Predictor",
    page_icon="🔮",
)

# --- Caching the Model and Preprocessing Objects ---
@st.cache_data
def setup_and_train_model():
    """
    Performs all data loading, cleaning, feature engineering, encoding, scaling,
    and model training. Caches the final objects needed for prediction.
    """
    # 1. LOAD DATA
    data = pd.read_csv('./cleanedData.csv')
    # Add this line
    data.drop(['exterior_color', 'interior_color'], axis=1, inplace=True)
    
    # 2. FEATURE ENGINEERING (This must be identical for training and prediction)
    # --- Engine ---
    data['engine_liters'] = data['engine'].str.extract(r'(\d\.?\d*)L', flags=re.IGNORECASE).astype(float)
    data['is_turbo'] = data['engine'].str.contains('Turbo', case=False, na=False).astype(int)
    data['is_hybrid'] = data['engine'].str.contains('Hybrid', case=False, na=False).astype(int)
    data['valve_count'] = data['engine'].str.extract(r'(\d{1,2})V', flags=re.IGNORECASE).astype(float)
    data['engine_liters'] = data['engine_liters'].fillna(data['engine_liters'].median())
    data['valve_count'] = data['valve_count'].fillna(data['valve_count'].median())

    # --- Transmission ---
    data['gears'] = data['transmission'].str.extract(r'(\d{1,2})-Speed', flags=re.IGNORECASE).astype(float)
    conditions = [
        data['transmission'].str.contains('CVT', case=False, na=False),
        data['transmission'].str.contains('Automatic', case=False, na=False)
    ]
    choices = ['CVT', 'Automatic']
    data['transmission_type'] = np.select(conditions, choices, default='Other')
    data['gears'] = data['gears'].fillna(data['gears'].median())
    
    # --- Model & Trim (Grouping rare categories) ---
    model_counts = data['model'].value_counts()
    trim_counts = data['trim'].value_counts()
    rare_models = model_counts[model_counts < 10].index
    rare_trims = trim_counts[trim_counts < 5].index
    data['model_cleaned'] = data['model'].replace(rare_models, 'Other')
    data['trim_cleaned'] = data['trim'].replace(rare_trims, 'Other')

    # --- Age ---
    current_year = 2025 # Use a fixed year consistent with training
    data['age'] = current_year - data['year']
    
    # --- Drop original and unused columns ---
    data.drop(['engine', 'transmission', 'model', 'trim'], axis=1, inplace=True)
    
    # 3. LABEL ENCODING
    categorical_cols = [
        'make', 'fuel', 'body', 'drivetrain', 
        'model_cleaned', 'trim_cleaned', 'transmission_type'
    ]
    
    # Store fitted encoders to transform user input later
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
        
    # 4. MODEL TRAINING
    X = data.drop('price', axis=1)
    y = data['price']
    
    # Save feature column order for prediction
    feature_columns = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=300, max_features=0.24, random_state=42, oob_score=True, max_depth=20, min_samples_leaf=1, min_samples_split=2)
    model.fit(X_scaled, y)
    
    # 5. PREPARE UI DATA
    # Load original cleaned data again to get text values for dropdowns
    ui_data = pd.read_csv('./cleanedData.csv')
    make_to_models = ui_data.groupby('make')['model'].unique().apply(list).to_dict()
    
    # Dictionary to return all necessary objects
    artifacts = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "label_encoders": label_encoders,
        "current_year": current_year,
        "rare_models": rare_models,
        "rare_trims": rare_trims,
        "ui_data": ui_data,
        "make_to_models": make_to_models
    }
    return artifacts

# --- Load all artifacts from the setup function ---
artifacts = setup_and_train_model()
model = artifacts["model"]
scaler = artifacts["scaler"]
FEATURE_COLUMNS = artifacts["feature_columns"]
label_encoders = artifacts["label_encoders"]
CURRENT_YEAR = artifacts["current_year"]
rare_models = artifacts["rare_models"]
rare_trims = artifacts["rare_trims"]
ui_data = artifacts["ui_data"]
make_to_models = artifacts["make_to_models"]

# --- Prediction Function ---
def predict_price(input_data):
    """
    Takes a dictionary of user inputs, applies the full preprocessing pipeline,
    and returns a price prediction.
    """
    df = pd.DataFrame([input_data])
    
    # Apply the same feature engineering steps
    df['engine_liters'] = df['engine'].str.extract(r'(\d\.?\d*)L', flags=re.IGNORECASE).astype(float).fillna(ui_data['engine'].str.extract(r'(\d\.?\d*)L', flags=re.IGNORECASE).astype(float).median())
    df['is_turbo'] = df['engine'].str.contains('Turbo', case=False, na=False).astype(int)
    df['is_hybrid'] = df['engine'].str.contains('Hybrid', case=False, na=False).astype(int)
    df['valve_count'] = df['engine'].str.extract(r'(\d{1,2})V', flags=re.IGNORECASE).astype(float).fillna(ui_data['engine'].str.extract(r'(\d{1,2})V', flags=re.IGNORECASE).astype(float).median())
    df['gears'] = df['transmission'].str.extract(r'(\d{1,2})-Speed', flags=re.IGNORECASE).astype(float).fillna(ui_data['transmission'].str.extract(r'(\d{1,2})-Speed', flags=re.IGNORECASE).astype(float).median())
    df['transmission_type'] = np.select([df['transmission'].str.contains('CVT', case=False, na=False), df['transmission'].str.contains('Automatic', case=False, na=False)], ['CVT', 'Automatic'], default='Other')
    df['model_cleaned'] = df['model'].apply(lambda x: 'Other' if x in rare_models else x)
    df['trim_cleaned'] = df['trim'].apply(lambda x: 'Other' if x in rare_trims else x)
    df['age'] = CURRENT_YEAR - df['year']
    
    df.drop(['engine', 'transmission', 'model', 'trim'], axis=1, inplace=True)
    
    # Apply fitted label encoders
    for col, le in label_encoders.items():
        # Handle unseen values by mapping them to a known category, e.g., the first one
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])
        
    # Align columns
    df_aligned = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    
    # Scale and predict
    X_input_scaled = scaler.transform(df_aligned)
    prediction = model.predict(X_input_scaled)
    
    return round(float(prediction[0]), 2)

# --- Streamlit User Interface ---
st.title("🚗 Vehicle Price Prediction Model")
st.sidebar.success("1. Choose Vehicle Options\n2. Click 'Predict'")
st.sidebar.divider()
st.sidebar.header("Your Selections:")

col1, col2 = st.columns([1, 1.5])

with col2:
    st.header("Choose Vehicle Features")
    
    # --- Dynamic and Dependent Dropdowns ---
    make = st.selectbox("Make", options=list(make_to_models.keys()), index=0)
    available_models = make_to_models[make]
    
    # <<< FIX 1: Rename this variable to avoid collision >>>
    model_choice = st.selectbox("Model", options=available_models, index=0)
    
    year_options = sorted(ui_data['year'].unique(), reverse=True)
    year = st.selectbox("Year", options=year_options, index=0)
    
    cylinders = st.selectbox("Cylinders", options=sorted(ui_data['cylinders'].unique()), index=2)
    fuel = st.selectbox("Fuel", options=ui_data['fuel'].unique(), index=0)
    body = st.selectbox("Body", options=ui_data['body'].unique(), index=0)
    doors = st.selectbox("Doors", options=sorted(ui_data['doors'].unique()), index=2)
    drivetrain = st.selectbox("Drivetrain", options=ui_data['drivetrain'].unique(), index=0)
    
    # Add new inputs
    engine = st.selectbox("Engine", options=ui_data['engine'].unique(), index=0)
    transmission = st.selectbox("Transmission", options=ui_data['transmission'].unique(), index=0)
    trim = st.selectbox("Trim", options=ui_data['trim'].unique(), index=0)
    
    mileage = st.slider("Mileage", 0, 100000, 5000, step=100)

# Display user choices in the sidebar
selections = {
    "Make": make, 
    "Model": model_choice, # <<< FIX 2: Use the new variable name here >>>
    "Year": year, 
    "Cylinders": cylinders,
    "Fuel": fuel, 
    "Body": body, 
    "Doors": doors, 
    "Drivetrain": drivetrain,
    "Engine": engine, 
    "Transmission": transmission, 
    "Trim": trim, 
    "Mileage": mileage
}

for label, value in selections.items():
    st.sidebar.write(f"**{label}:** {value}")

with col1:
    st.header("Predicted Price")
    if st.button("Predict 🔮", use_container_width=True):
        input_data = {
            'make': make, 
            'model': model_choice, # <<< FIX 3: And use the new variable name here >>>
            'year': year, 
            'cylinders': cylinders, 
            'fuel': fuel, 
            'mileage': mileage, 
            'body': body, 
            'doors': doors, 
            'drivetrain': drivetrain,
            'engine': engine, 
            'transmission': transmission, 
            'trim': trim
        }
        price = predict_price(input_data)
        st.metric("Estimated Value", f"$ {price:,.2f}")
    else:
        st.metric("Estimated Value", "$ 0.00")