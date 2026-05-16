import streamlit as st
import joblib
import numpy as np

# Cargar modelo y scaler
modelo = joblib.load("modelos/random_forest_model.pkl")
scaler = joblib.load("modelos/scaler_model.pkl")

st.title("🩺 Predicción de Diabetes")

st.write("Ingrese los datos del paciente y presione el botón Predecir.:")

# Entradas del usuario
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)

glucose = st.number_input("Glucose", min_value=0.0, step=1.0)

bloodpressure = st.number_input("Blood Pressure", min_value=0.0, step=1.0)

skinthickness = st.number_input("Skin Thickness", min_value=0.0, step=1.0)

insulin = st.number_input("Insulin", min_value=0.0, step=1.0)

bmi = st.number_input("BMI", min_value=0.0, step=0.1, format="%.1f")

dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    step=0.001,
    format="%.3f"
)

age = st.number_input("Age", min_value=1, max_value=120, step=1)

# Botón de predicción
if st.button("Predecir"):

    datos = np.array([[
        pregnancies,
        glucose,
        bloodpressure,
        skinthickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    # Escalar datos
    datos = scaler.transform(datos)

    # Predicción
    prediccion = modelo.predict(datos)

    if prediccion[0] == 1:
        st.error("La persona tiene diabetes")
    else:
        st.success("La persona NO tiene diabetes")

# Mis Datos
st.write("Nombre: RICARDO MARTIN CESPEDES PINO")
st.write("Código ISIL: 71240691")

# Link de Colab
st.markdown("https://colab.research.google.com/drive/11oBppWOzXTqCWIlwP8E6jjoedXLwbvdF?usp=sharing")

