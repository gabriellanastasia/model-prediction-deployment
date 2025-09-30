import streamlit as st
import joblib
import pandas as pd

def run_ml_app():
    # Load the saved model and scaler
    try:
        best_model = joblib.load('linear_regression_model.pkl')
        scaler = joblib.load('scaler.pkl')
        st.success("Model and scaler loaded successfully.")
    except FileNotFoundError:
        st.error("Error: Model or scaler file not found. Please ensure the .pkl files are in the same directory.")
        best_model, scaler = None, None

    # Define expected features (same as training)
    training_columns = [
        'mileage_kmpl', 'engine_cc', 'owner_count', 'accidents_reported',
        'car_age', 'fuel_type_Electric', 'fuel_type_Petrol',
        'brand_Chevrolet', 'brand_Ford', 'brand_Honda', 'brand_Hyundai',
        'brand_Kia', 'brand_Nissan', 'brand_Tesla', 'brand_Toyota',
        'brand_Volkswagen', 'transmission_Manual', 'color_Blue',
        'color_Gray', 'color_Red', 'color_Silver', 'color_White',
        'service_history_Partial', 'service_history_Unknown',
        'insurance_valid_Yes'
    ]

    # UI
    st.title("🚗 Used Car Price Prediction")
    st.write("Enter car details below to predict its price:")

    fuel_types = ['Petrol', 'Diesel', 'Electric']
    brands = ['Chevrolet', 'Honda', 'BMW', 'Hyundai', 'Nissan', 'Kia',
              'Tesla', 'Toyota', 'Volkswagen', 'Ford']
    transmissions = ['Manual', 'Automatic']
    colors = ['White', 'Black', 'Blue', 'Red', 'Gray', 'Silver']
    service_histories = ['Unknown', 'Full', 'Partial']
    insurance_validity = ['Yes', 'No']

    col1, col2 = st.columns(2)
    with col1:
        make_year = st.number_input("Make Year", min_value=1990, max_value=2023, value=2015, step=1)
        mileage_kmpl = st.number_input("Mileage (kmpl)", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
        engine_cc = st.number_input("Engine Capacity (cc)", min_value=500, max_value=6000, value=1500, step=100)
        owner_count = st.number_input("Owner Count", min_value=1, max_value=5, value=2, step=1)
        accidents_reported = st.number_input("Accidents Reported", min_value=0, max_value=10, value=0, step=1)

    with col2:
        fuel_type = st.selectbox("Fuel Type", fuel_types)
        brand = st.selectbox("Brand", brands)
        transmission = st.selectbox("Transmission", transmissions)
        color = st.selectbox("Color", colors)
        service_history = st.selectbox("Service History", service_histories)
        insurance_valid = st.selectbox("Insurance Valid", insurance_validity)

    if st.button("Predict Price"):
        if best_model and scaler:
            # DataFrame
            input_data = pd.DataFrame({
                'make_year': [make_year],
                'mileage_kmpl': [mileage_kmpl],
                'engine_cc': [engine_cc],
                'owner_count': [owner_count],
                'accidents_reported': [accidents_reported],
                'fuel_type': [fuel_type],
                'brand': [brand],
                'transmission': [transmission],
                'color': [color],
                'service_history': [service_history],
                'insurance_valid': [insurance_valid]
            })

            # Feature engineering
            CURRENT_YEAR = 2023
            input_data["car_age"] = CURRENT_YEAR - input_data["make_year"].astype(int)
            input_data = input_data.drop(columns=["make_year"])

            # One-hot encode
            input_data_encoded = pd.get_dummies(
                input_data,
                columns=["fuel_type", "brand", "transmission", "color", "service_history", "insurance_valid"],
                drop_first=True
            )

            # Align columns
            input_data_aligned = input_data_encoded.reindex(columns=training_columns, fill_value=0)

            # Scale
            num_features = ["mileage_kmpl", "engine_cc", "owner_count", "accidents_reported", "car_age"]
            input_data_aligned[num_features] = scaler.transform(input_data_aligned[num_features])

            # Predict
            predicted_price = best_model.predict(input_data_aligned)
            st.success(f"Predicted Car Price: ${predicted_price[0]:,.2f}")
        else:
            st.warning("Model or scaler not loaded. Cannot make prediction.")
