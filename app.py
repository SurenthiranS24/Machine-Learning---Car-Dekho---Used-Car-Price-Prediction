import pickle
import pandas as pd
import streamlit as slt
import numpy as np

# Page configuration
slt.set_page_config(page_title="Cardheko-Price Prediction 🚗", layout="wide")

# Header
slt.markdown("<h1 style='text-align: center; color: orange;'>🚗 Cardheko Price Prediction</h1>", unsafe_allow_html=True)
slt.markdown("#### *Get an estimate of the resale value of your car in just a few clicks!*")
slt.write("---")  # Divider line

# Load data
df = pd.read_csv("final_model.csv")

# Sidebar setup for better navigation
slt.sidebar.header("Select Car Specifications")

# Sidebar Inputs
Ft = slt.sidebar.selectbox("Fuel type", ['Petrol', 'Diesel', 'Lpg', 'Cng', 'Electric'])
Bt = slt.sidebar.selectbox("Body type", ['Hatchback', 'SUV', 'Sedan', 'MUV', 'Coupe', 'Minivans',
                                         'Convertibles', 'Hybrids', 'Wagon', 'Pickup Trucks'])
Tr = slt.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])
Owner = slt.sidebar.selectbox("Owner", [0, 1, 2, 3, 4, 5])
brand = slt.sidebar.selectbox("Brand", options=df['Brand'].unique())
Model = slt.sidebar.selectbox("Model Year", [2015, 2018, 2014, 2020, 2017, 2021, 2019, 2022, 2016, 2011, 2009,
                                             2013, 2010, 2008, 2006, 2012, 2005, 2007, 2023, 1998, 2004, 2003,
                                             2001, 2002, 2000, 1985, 1997, 1999])
IV = slt.sidebar.selectbox("Insurance Validity", ['Third Party insurance', 'Comprehensive', 'Third Party',
                                                  'Zero Dep', '2', '1', 'Not Available'])
Km = slt.sidebar.slider("Kilometers Driven", min_value=100, max_value=100000, step=1000)
ML = slt.sidebar.slider("Mileage", min_value=5, max_value=150, step=1)
EG = slt.sidebar.slider("Engine", min_value=70, max_value=5500, step=1)
seats = slt.sidebar.selectbox("Seats", [5, 7, 8, 6, 4, 10, 9, 2])
color = slt.sidebar.selectbox("Color", options=df['Color'].unique())
city = slt.sidebar.selectbox("City", options=df['City'].unique())

# Main page layout with two columns for results and data preview
col1, col2 = slt.columns(2)

with col1:
    slt.subheader("📝 Selected Car Details")
    slt.markdown("Here are the details of the car you selected:")
    slt.write(f"**Fuel type**: {Ft}")
    slt.write(f"**Body type**: {Bt}")
    slt.write(f"**Transmission**: {Tr}")
    slt.write(f"**Owner count**: {Owner}")
    slt.write(f"**Brand**: {brand}")
    slt.write(f"**Model Year**: {Model}")
    slt.write(f"**Insurance**: {IV}")
    slt.write(f"**Kilometers Driven**: {Km}")
    slt.write(f"**Mileage**: {ML}")
    slt.write(f"**Engine**: {EG}")
    slt.write(f"**Seats**: {seats}")
    slt.write(f"**Color**: {color}")
    slt.write(f"**City**: {city}")

with col2:
    slt.subheader("💡 Prediction")
    slt.markdown("Click the button to predict the car price:")

    if slt.button("Predict Car Price"):
        # Load the saved model, scaler, and label encoders
        with open('Randomforest_regression.pkl', 'rb') as files:
            final_model = pickle.load(files)

        with open('std_scaler.pkl', 'rb') as files_std:
            scaler = pickle.load(files_std)

        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)

        # Prepare input data
        new_df = pd.DataFrame({
            'Fuel type': [Ft],
            'body type': [Bt],
            'transmission': [Tr],
            'ownerNo': [Owner],
            'Brand': [brand],
            'modelYear': [Model],
            'Insurance Validity': [IV],
            'Kms Driven': [Km],
            'Mileage': [ML],
            'Engine': [EG],
            'Seats': [seats],
            'Color': [color],
            'City': [city]
        })

        # Apply LabelEncoder to categorical columns
        for col in new_df.columns:
            if new_df[col].dtype == 'object' or new_df[col].dtype.name == 'category':
                le = label_encoders[col]
                new_df[col] = le.transform(new_df[col])

        # Scale the data
        scaling = scaler.transform(new_df)

        # Model prediction
        prediction = final_model.predict(scaling)

        # Display the predicted result
        slt.write(f"Cardheko predicted car price is: **₹ {round(prediction[0], 2)}**")

# Footer with spacing and additional details
slt.markdown("---")
slt.markdown("Cardheko-Price Prediction Created by Surenthiran S - Data Science Enthusiast 🚀")
