import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Car Price Predictor | Basak Tamer", 
    page_icon="🚗", 
    layout="centered"
)

# --- 2. LOAD MODELS & ASSETS ---
# Using @st.cache_resource to load these into memory only once
@st.cache_resource
def load_assets():
    try:
        # These filenames must match your uploaded files exactly
        model = joblib.load('car_price_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('features.pkl')
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

model, scaler, feature_names = load_assets()

# --- 3. FEATURE ENGINEERING HELPERS ---
def clean_engine(engine_str):
    """Extracts HP and Liter values from raw engine strings"""
    hp = re.search(r'(\d+\.?\d*)HP', str(engine_str))
    liter = re.search(r'(\d+\.?\d*)L', str(engine_str))
    hp_val = float(hp.group(1)) if hp else 240.0
    liter_val = float(liter.group(1)) if liter else 2.0
    return hp_val, liter_val

def clean_color(color_str):
    """Categorizes paint as premium based on keywords"""
    c = str(color_str).upper()
    return 1 if any(x in c for x in ['METALLIC', 'PEARL', 'COAT', 'MATTE']) else 0

# --- 4. USER INTERFACE ---
st.title("🚗 Used Car Price Predictor")
st.markdown("""
Welcome to my car price prediction app. This model uses **Gradient Boosting Regression** 
to estimate vehicle values based on technical specifications[cite: 1].
""")

if model is not None:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            brand = st.selectbox("Brand", ["Ford", "BMW", "Mercedes-Benz", "Toyota", "Audi", "Porsche", "Chevrolet"])
            year = st.number_input("Model Year", 1990, 2024, 2018)
            milage = st.number_input("Mileage (mi)", 0, 500000, 50000)
            
        with col2:
            engine_desc = st.text_input("Engine (e.g., 300HP 3.0L)", "240.0HP 2.0L")
            ext_color = st.text_input("Exterior Color", "Black Metallic")
            transmission = st.selectbox("Transmission", ["Automatic", "Manual", "CVT"])

        submit = st.form_submit_button("Predict Estimated Price")

    # --- 5. PREDICTION LOGIC ---
    if submit:
        # A. Feature Extraction[cite: 1]
        hp, liter = clean_engine(engine_desc)
        premium_paint = clean_color(ext_color)
        age = 2024 - year
        milage_per_year = milage / (age + 1)
        
        # B. Create Input DataFrame (matching feature order exactly)[cite: 1]
        input_df = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
        
        # C. Map Numerical Values[cite: 1]
        mapping = {
            'model_year': year,
            'milage': milage,
            'milage_per_year': milage_per_year,
            'engine_hp': hp,
            'engine_liter': liter,
            'is_premium_paint': premium_paint
        }
        
        for col, val in mapping.items():
            if col in input_df.columns:
                input_df[col] = val
        
        # D. Map Categorical Dummies[cite: 1]
        brand_col = f"brand_{brand}"
        trans_col = f"transmission_{transmission}"
        
        if brand_col in input_df.columns:
            input_df[brand_col] = 1
        if trans_col in input_df.columns:
            input_df[trans_col] = 1

        # E. Scaling and Prediction[cite: 1]
        try:
            # Important: Transform data using the loaded scaler[cite: 1]
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            
            st.divider()
            st.subheader(f"Estimated Price: :green[${prediction:,.2f}]")
            st.balloons()
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("This often happens if input columns don't match the model's training features.")
else:
    st.warning("Model assets are missing. Please upload your .pkl files to the Space.")