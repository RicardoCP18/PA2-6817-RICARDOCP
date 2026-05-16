import streamlit as st
import joblib
import numpy as np

# Cargar modelo y scaler
modelo = joblib.load("modelos/random_forest_model.pkl")
scaler = joblib.load("modelos/scaler_model.pkl")

st.title("Predicción de Diabetes")

st.write("Aplicación desarrollada para PA2")

# Entradas del usuario
pregnancies = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
bloodpressure = st.number_input("Blood Pressure")
skinthickness = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

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
st.markdown("https://colab.research.google.com/drive/11oBppWOzXTqCWIlwP8E6jjoedXLwbvdF#scrollTo=029Io0Dn6mJj")

