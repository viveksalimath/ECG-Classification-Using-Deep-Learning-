
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

st.title("ECG Classification Demo")

uploaded_file = st.file_uploader("Upload ECG CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    X_input = data.values
    X_input = (X_input - X_input.mean()) / X_input.std()
    X_input = X_input.reshape(X_input.shape[0], X_input.shape[1], 1)
    
    model = load_model('ecg_hybrid_model.h5')
    predictions = model.predict(X_input)
    predicted_classes = np.argmax(predictions, axis=1)
    
    st.write("Predicted Classes:")
    st.write(predicted_classes)
