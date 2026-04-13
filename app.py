import streamlit as st 
import pickle 
import numpy as np

#Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("Cardio Fitness Income Prediction")

age = st.slider("Age", 18, 25, 60)
education = st.slider("Education Level", 10, 20, 15)
usage = st.slider("Usage", 1, 5, 3)
fitness = st.slider("Fitness Level", 1, 5, 7)
miles = st.slider("Miles Run", 0, 300, 1000)

if st.button("Predict Income"):
    data = np.array([[age, education, usage, fitness, miles]])
    prediction = model.predict(data)
    st.success(f"Predict Income: {prediction[0]:.2f}")
    