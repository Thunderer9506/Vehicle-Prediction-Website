import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re

st.set_page_config(page_title="Vehicle Price Predictor", page_icon="🔮", layout="wide")

# --- Load pre-trained model ---
@st.cache_resource
def load_artifacts():
    with open('artifacts.pkl', 'rb') as f:
        return pickle.load(f)

artifacts = load_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]
label_encoders = artifacts["label_encoders"]
FEATURE_COLUMNS = artifacts["feature_columns"]
CURRENT_YEAR = artifacts["current_year"]
rare_models = artifacts["rare_models"]
rare_trims = artifacts["rare_trims"]
evaluation = artifacts["metrics"]

# --- UI Header ---
st.markdown("<h1 style='text-align:center;'>🚗 Vehicle Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:1.2rem;'>Get an instant estimate for your car’s value</p>", unsafe_allow_html=True)

# --- Sidebar: Model Description ---
with st.sidebar:
    st.info("Model pre-trained and cached for instant predictions.")
    st.divider()
    st.header("📝 Model Description")
    st.markdown(
        """
        <div style='font-size: 0.9rem; line-height: 1.4;'>
        1️⃣ <b>Algorithm:</b> Uses a <b>Random Forest Regressor</b> — an ensemble of decision trees that enhances accuracy and reduces overfitting.  
        <br><br>
        2️⃣ <b>Training Data:</b> Built on a cleaned dataset including features such as make, model, year, drivetrain, body, and engine specs.  
        <br><br>
        3️⃣ <b>Feature Engineering:</b> Extracts engine details (liters, turbo/hybrid, valves) and transmission type with categorical encoding for robust predictions.  
        <br><br>
        4️⃣ <b>Performance:</b> Achieved <b>R² > 0.9</b> with low MAE and RMSE, ensuring high reliability.  
        <br><br>
        5️⃣ <b>Efficiency:</b> Fully <b>pre-trained and cached</b> for instant predictions without retraining.  
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Prediction Function ---
def predict_price(df):
    df['engine_liters'] = df['engine'].str.extract(r'(\d\.?\d*)L', flags=re.IGNORECASE).astype(float).fillna(2.0)
    df['is_turbo'] = df['engine'].str.contains('Turbo', case=False, na=False).astype(int)
    df['is_hybrid'] = df['engine'].str.contains('Hybrid', case=False, na=False).astype(int)
    df['valve_count'] = df['engine'].str.extract(r'(\d{1,2})V', flags=re.IGNORECASE).astype(float).fillna(16)
    df['gears'] = df['transmission'].str.extract(r'(\d{1,2})-Speed', flags=re.IGNORECASE).astype(float).fillna(6)
    df['transmission_type'] = np.select(
        [df['transmission'].str.contains('CVT', case=False, na=False),
         df['transmission'].str.contains('Automatic', case=False, na=False)],
        ['CVT', 'Automatic'], default='Other'
    )
    df['model_cleaned'] = df['model'].apply(lambda x: 'Other' if x in rare_models else x)
    df['trim_cleaned'] = df['trim'].apply(lambda x: 'Other' if x in rare_trims else x)
    df['age'] = CURRENT_YEAR - df['year']

    df.drop(['engine', 'transmission', 'model', 'trim'], axis=1, inplace=True)
    for col, le in label_encoders.items():
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])

    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return round(float(model.predict(scaler.transform(df))[0]), 2)

# --- UI Tabs ---
col1, col2 = st.columns([1, 1.5])
ui_data = pd.read_csv('cleanedData.csv')
make_to_models = ui_data.groupby('make')['model'].unique().apply(list).to_dict()

with col2:
    st.header("⚙️ Choose Vehicle Features")
    featurecol1, featurecol2 = st.columns(2)
    with featurecol1:
        make = st.selectbox("Make", options=list(make_to_models.keys()))
        model_choice = st.selectbox("Model", options=make_to_models[make])
        year = st.selectbox("Year", options=sorted(ui_data['year'].unique(), reverse=True))
        mileage = st.slider("Mileage (in thousands)", 0, 25, 7)
        engine = st.selectbox("Engine", options=ui_data['engine'].unique())
        transmission = st.selectbox("Transmission", options=ui_data['transmission'].unique())
    with featurecol2:
        cylinders = st.selectbox("Cylinders", options=sorted(ui_data['cylinders'].unique()))
        body = st.selectbox("Body", options=ui_data['body'].unique())
        drivetrain = st.selectbox("Drivetrain", options=ui_data['drivetrain'].unique())
        fuel = st.selectbox("Fuel", options=ui_data['fuel'].unique())
        doors = st.selectbox("Doors", options=sorted(ui_data['doors'].unique()))
        trim = st.selectbox("Trim", options=ui_data['trim'].unique())

with col1:
    st.subheader("💵 Predicted Value")
    if st.button("Predict 🔮", use_container_width=True):
        input_data = {
            'make': make, 'model': model_choice, 'year': year, 'cylinders': cylinders,
            'fuel': fuel, 'mileage': mileage, 'body': body, 'doors': doors,
            'drivetrain': drivetrain, 'engine': engine, 'transmission': transmission, 'trim': trim
        }
        df = pd.DataFrame([input_data])
        price = predict_price(df)
        st.metric("Estimated Price", f"$ {price:,.2f}")
    else:
        st.metric("Estimated Price", "$ 0.00")
        
    with st.container():
        st.header("📊 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", f"{evaluation['R² Score']:.3f}")
            st.metric("MAE", f"{evaluation['MAE']:.2f}")
        with col2:
            st.metric("RMSE", f"{evaluation['RMSE']:.2f}")
            st.metric("OOB Score", f"{evaluation['OOB Score']:.2f}")

st.markdown("---")
st.caption("© 2025 Vehicle Price Predictor | Fast predictions powered by pre-trained RandomForestRegressor 🌟")
