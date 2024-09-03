import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = load_model('land_mine_detection_dnn_model.h5')

# Create a StandardScaler instance (assuming the model was trained with scaled data)
scaler = StandardScaler()

# Set up the title and description for the Streamlit app
st.title("Land Mine Detection using DNN")
st.write("""
    This application classifies land mines based on the input features using a Deep Neural Network model.
    Please provide the necessary input values, and the model will predict the type of land mine.
""")

# Input fields for the user to enter data
V = st.number_input("Voltage (V)", value=0.0)
H = st.number_input("Height from ground (cm)", value=0.0)
S = st.selectbox("Soil Type", ["dry and sandy", "dry and humus", "dry and limy", "humid and sandy", "humid and humus", "humid and limy"])

# Convert soil type to numerical value
soil_type_mapping = {
    "dry and sandy": 0,
    "dry and humus": 1,
    "dry and limy": 2,
    "humid and sandy": 3,
    "humid and humus": 4,
    "humid and limy": 5
}

S = soil_type_mapping[S]

# Create a DataFrame for the input values
input_data = pd.DataFrame([[V, H, S]], columns=['V', 'H', 'S'])

# Assuming the scaler was fitted on the training data
# Here we scale input data for consistency
scaled_input = scaler.fit_transform(input_data)  # Replace fit_transform with transform if you already have a saved scaler

# Make predictions
prediction = model.predict(scaled_input)
predicted_class = np.argmax(prediction, axis=1)

# Define class names (Adjust according to your dataset's classes)
mine_classes = {
    0: "Mine Type A",
    1: "Mine Type B",
    2: "Mine Type C",
    3: "Mine Type D",
    4: "Mine Type E"
}

# Display the result
st.write(f"Predicted Mine Type: {mine_classes[predicted_class[0]]}")
