import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import warnings

# Suppress version mismatch warnings
warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Load model, scaler, and column list
@st.cache_resource
def load_artifacts():
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("columns.pkl", "rb") as f:
        columns = pickle.load(f)
    return model, scaler, columns

try:
    model, scaler, columns = load_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Header Section
st.title("❤️ Heart Disease Prediction App")
st.markdown(
    """
    Enter patient clinical parameters to assess the risk of heart disease 
    using the trained **K-Nearest Neighbors (KNN)** machine learning model.
    """
)
st.write("---")

# Layout Form in Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Patient Demographics & Vitals")
    age = st.slider("Age (years)", min_value=20, max_value=100, value=50, step=1)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120, step=1)
    cholesterol = st.number_input("Serum Cholesterol (mg/dL)", min_value=0, max_value=600, value=200, step=1)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=["No (<= 120 mg/dL)", "Yes (> 120 mg/dL)"])

with col2:
    st.subheader("🩺 Clinical & Exercise Tests")
    chest_pain = st.selectbox(
        "Chest Pain Type",
        options=[
            "Asymptomatic (ASY)",
            "Atypical Angina (ATA)",
            "Non-Anginal Pain (NAP)",
            "Typical Angina (TA)"
        ]
    )
    resting_ecg = st.selectbox(
        "Resting Electrocardiogram (ECG)",
        options=[
            "Normal",
            "ST-T Wave Abnormality (ST)",
            "Left Ventricular Hypertrophy (LVH)"
        ]
    )
    max_hr = st.slider("Maximum Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=140, step=1)
    exercise_angina = st.selectbox("Exercise-Induced Angina", options=["No", "Yes"])
    oldpeak = st.number_input("Oldpeak (ST Depression in mm)", min_value=-3.0, max_value=7.0, value=0.0, step=0.1)
    st_slope = st.selectbox(
        "ST Slope",
        options=[
            "Upsloping (Up)",
            "Flat",
            "Downsloping (Down)"
        ]
    )

st.write("---")

# Prediction Button
if st.button("🔍 Predict Heart Disease Risk", use_container_width=True, type="primary"):
    # Encoding inputs to match trained model columns
    input_data = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': 1 if "Yes" in fasting_bs else 0,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_M': 1 if sex == "Male" else 0,
        'ChestPainType_ATA': 1 if "ATA" in chest_pain else 0,
        'ChestPainType_NAP': 1 if "NAP" in chest_pain else 0,
        'ChestPainType_TA': 1 if "TA" in chest_pain else 0,
        'RestingECG_Normal': 1 if resting_ecg == "Normal" else 0,
        'RestingECG_ST': 1 if "ST" in resting_ecg else 0,
        'ExerciseAngina_Y': 1 if exercise_angina == "Yes" else 0,
        'ST_Slope_Flat': 1 if st_slope == "Flat" else 0,
        'ST_Slope_Up': 1 if "Up" in st_slope else 0
    }

    # Convert to DataFrame with correct column order
    input_df = pd.DataFrame([input_data])[columns]

    try:
        # Scale numerical features
        scaled_input = scaler.transform(input_df)

        # Run model prediction
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]

        st.subheader("📊 Prediction Result")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if prediction == 1:
                st.error("⚠️ **High Risk: Heart Disease Detected**")
                st.write("The model suggests an elevated likelihood of heart disease.")
            else:
                st.success("✅ **Low Risk: Normal / Healthy**")
                st.write("The model suggests a low likelihood of heart disease.")

        with res_col2:
            st.metric(
                label="Heart Disease Probability",
                value=f"{prediction_proba[1] * 100:.1f}%"
            )
            st.progress(float(prediction_proba[1]))

    except Exception as err:
        st.error(f"Error during prediction: {err}")
