# ❤️ Heart Diseases Prediction Web App

A machine learning web application built with **Streamlit** and **Scikit-Learn** that predicts the risk of cardiovascular disease based on clinical patient parameters.

---

## 📌 Project Overview

This application utilizes a trained **K-Nearest Neighbors (KNN)** model and a **StandardScaler** to assess a patient's risk of heart disease in real-time.

### Features Considered:
- **Age**: Patient's age in years
- **Sex**: Biological sex (Male / Female)
- **RestingBP**: Resting blood pressure (mm Hg)
- **Cholesterol**: Serum cholesterol (mg/dL)
- **FastingBS**: Fasting blood sugar (> 120 mg/dL)
- **ChestPainType**: Asymptomatic (ASY), Atypical Angina (ATA), Non-Anginal Pain (NAP), Typical Angina (TA)
- **RestingECG**: Normal, ST-T wave abnormality (ST), Left ventricular hypertrophy (LVH)
- **MaxHR**: Maximum heart rate achieved during exercise (bpm)
- **ExerciseAngina**: Exercise-induced angina (Yes / No)
- **Oldpeak**: ST depression induced by exercise relative to rest (mm)
- **ST_Slope**: Slope of the peak exercise ST segment (Up, Flat, Down)

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- pip package manager

### 2. Clone the Repository
```bash
git clone https://github.com/KevinMendapara/Heart-Diseases-Prediction.git
cd Heart-Diseases-Prediction
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
├── app.py              # Main Streamlit web application
├── KNN_heart.pkl       # Trained K-Nearest Neighbors model
├── scaler.pkl          # Standard scaler for numerical features
├── columns.pkl         # Feature column names and order
├── requirements.txt    # Python dependencies
├── .gitignore          # Ignored files (virtual environments, cache)
└── README.md           # Documentation
```

---

## 👤 Author
- **Kevin Mendapara** - [GitHub Profile](https://github.com/KevinMendapara)
