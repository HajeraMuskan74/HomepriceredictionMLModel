import streamlit as st
import pickle
import json
import numpy as np

# Load model and columns
with open("Model/home_price_model.pickle", "rb") as f:
    model = pickle.load(f)

with open("Model/columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]
    locations = data_columns[3:]  # Assuming first 3 are sqft, bath, bhk

# Title
st.title("🏠 Bengaluru Home Price Predictor")

# Inputs
sqft = st.number_input("Total Square Feet", min_value=500)
bath = st.selectbox("Bathrooms", [1, 2, 3, 4])
bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])
location = st.selectbox("Location", sorted(locations))

# Prediction
if st.button("Estimate Price"):
    try:
        x = np.zeros(len(data_columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk
        if location in data_columns:
            loc_index = data_columns.index(location)
            x[loc_index] = 1

        price = round(model.predict([x])[0], 2)
        st.success(f"Estimated Price: ₹ {price} Lakhs")
    except Exception as e:
        st.error(f"Prediction failed: {e}")