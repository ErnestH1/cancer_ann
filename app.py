import streamlit as st
import numpy as np 
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

model=load_model('breast_cancer_model.h5')
scaler=StandardScaler()

st.title('Breast Cancer Prediction')

key_features=['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness']

inputs={}
for feature in key_features:
    inputs[feature] = st.number_input(feature.title(),0.0,100.0,10.0)

if st.button('Predict'):
    input_data=np.array([[inputs[feature]for feature in key_features]])
    prediction=model.predict(input_data)
    result='Malignant' if prediction[0][0]>0.5 else 'Benign'

    st.write(f"Prediction:  {result}")
    st.write(f"Confidence: {prediction[0][0]:.2f}")