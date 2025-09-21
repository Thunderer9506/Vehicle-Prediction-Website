# TODO: Re clean the data properly in jupyter notebook
# TODO: Change the hardcode values dynamic so that they change according to whats acctually is present in
#       the data
# TODO: Put some initial values in select box

# idx = 0
# engineDelete = []
# def func(x):
#     global idx
#     if ('<' in x):
#         engineDelete.append(idx)
#         print(idx)
#     idx += 1

# cleanedData['engine'].apply(func)


import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

cleanedData = pd.read_csv('./cleaned_dataset.csv')

# --- Page Configuration (should be the first st command) ---
st.set_page_config(
    page_title="Vehicle Price Predictor",
    page_icon="🔮",
)

# --- Caching the Model Loading and Training ---
@st.cache_data
def trainingModel():
    """
    Loads data, trains the RandomForestRegressor model, and returns the
    model, scaler, and feature columns. This function is cached for efficiency.
    """
    
    # Use a fixed year from training for consistent 'age' calculation
    CURRENT_YEAR = 2024
    
    scaler = StandardScaler()
    predictionData = cleanedData[['make', 'year', 'price', 'cylinders', 'fuel', 'mileage', 'body', 'doors', 'drivetrain']].copy()
    predictionData.loc[:, 'age'] = CURRENT_YEAR - predictionData['year']
    
    # One-hot encode categorical features
    encodedData = pd.get_dummies(predictionData, columns=['make', 'fuel', 'body','drivetrain'], drop_first=True)

    # Define features (X) and target (y)
    # Explicitly drop 'year' as 'age' is the feature we are using
    X = encodedData.drop(['price', 'year'], axis=1)
    y = encodedData['price']

    # Save feature column order for later use during prediction
    feature_columns = X.columns.tolist()

    X_train_scaled = scaler.fit_transform(X)

    # Initialize and train the model
    tunedregressor = RandomForestRegressor(n_estimators=100, max_features=0.57, random_state=0, oob_score=True)
    tunedregressor.fit(X_train_scaled, y)
    
    return tunedregressor, scaler, feature_columns, CURRENT_YEAR

# --- Load Model and necessary objects ---
model, scaler, FEATURE_COLUMNS, CURRENT_YEAR = trainingModel()


# --- Prediction Function (Now accepts arguments) ---
## <<< CHANGE 1: The function now takes all the user inputs as arguments.
def predictData(make, year, cylinders, fuel, mileage, body, doors, drivetrain):
    """
    Takes user inputs, preprocesses them, and returns a price prediction.
    """
    # Create a single-row DataFrame from the inputs
    data = {'make': make, 'year': year, 'cylinders': cylinders, 'fuel': fuel, 
            'mileage': mileage, 'body': body, 'doors': doors, 'drivetrain': drivetrain}
    df = pd.DataFrame([data])
    
    # Feature Engineering: Create 'age' feature consistent with training
    df['age'] = CURRENT_YEAR - df['year']

    # One-hot encode the new data
    encoded = pd.get_dummies(df, columns=['make', 'fuel', 'body','drivetrain'], drop_first=True)
    
    # Align columns with the training data, filling missing columns with 0
    encoded_aligned = encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # Scale the input data using the already-fitted scaler
    X_input_scaled = scaler.transform(encoded_aligned)

    # Make the prediction
    prediction = model.predict(X_input_scaled)
    
    return round(float(prediction[0]), 2)


# --- Streamlit User Interface ---
st.title("🚗 Vehicle Price Prediction Model")

# Sidebar for instructions and displaying user choices
st.sidebar.success("1. Choose Options\n" "2. Click on Predict Button")
st.sidebar.divider()
st.sidebar.header("Your Selections:")

# Main page layout
col1, col2 = st.columns([1, 1.5]) # Give more space to the inputs

# --- Input Widgets in the second column ---
with col2:
    st.header("Choose Vehicle Features")
    make = st.selectbox("Make", list(cleanedData['make'].unique()))
    year = st.selectbox("Year", [2025, 2024, 2023])
    cylinders = st.selectbox("Cylinders", [3, 4, 6, 8, 0]) # 0 for electric
    fuel = st.selectbox("Fuel", ['Gasoline', 'Diesel', 'Hybrid', 'Electric', 'E85 Flex Fuel',
                                 'PHEV Hybrid Fuel', 'Diesel (B20 capable)'])
    mileage = st.slider("Mileage", 0, 6000, 10, step=10) # Slider is better for numeric input
    body = st.selectbox("Body", ['SUV', 'Pickup Truck', 'Sedan', 'Passenger Van', 'Cargo Van',
                                 'Hatchback', 'Convertible', 'Minivan'])
    doors = st.selectbox("Doors", [2, 3, 4, 5])
    drivetrain = st.selectbox("Drivetrain", ['Four-wheel Drive', 'Rear-wheel Drive', 'All-wheel Drive',
                                             'Front-wheel Drive'])

# Display user choices in the sidebar
st.sidebar.write(f"**Make:** {make}")
st.sidebar.write(f"**Year:** {year}")
st.sidebar.write(f"**Cylinders:** {cylinders}")
st.sidebar.write(f"**Fuel:** {fuel}")
st.sidebar.write(f"**Mileage:** {mileage}")
st.sidebar.write(f"**Body:** {body}")
st.sidebar.write(f"**Doors:** {doors}")
st.sidebar.write(f"**Drivetrain:** {drivetrain}")

# --- Prediction Display and Button in the first column ---
with col1:
    st.header("Predicted Price")
    
    # Prediction button
    if st.button("Predict 🔮", use_container_width=True):
        ## <<< CHANGE 2: Pass the widget values directly to the function.
        price = predictData(make, year, cylinders, fuel, mileage, body, doors, drivetrain)
        st.metric("Estimated Value", f"$ {price:,.2f}")
    else:
        st.metric("Estimated Value", "$ 0.00")