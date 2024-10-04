import pickle
import pandas as pd
import streamlit as slt
import numpy as np

# Page configuration
slt.set_page_config(page_title="Cardheko Price Prediction", page_icon="🚗", layout="wide")

# Custom CSS for styling
slt.markdown(
    """
    <style>
    body {
        color: #fff;
        background-color: #2C2F33;
    }
    .css-18e3th9 {
        background-color: #2C2F33;
    }
    h1 {
        color: #FFA500 !important;
        text-align: center;
    }
    h2, h3 {
        color: #fff !important;
    }
    .stSelectbox, .stSlider {
        background-color: #202225;
        color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header section
slt.markdown("<h1>🚗 Cardheko Price Prediction</h1>", unsafe_allow_html=True)
slt.markdown("<h3 style='text-align: center;'>Accurate car resale value estimator!</h3>", unsafe_allow_html=True)
slt.write("---")

# Sidebar for car selection input
slt.sidebar.header("🚘 Select Car Specifications")

# Load data
df = pd.read_csv("final_model.csv")

# Sidebar inputs
Ft = slt.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Lpg', 'Cng', 'Electric'])
Bt = slt.sidebar.selectbox("Body Type", ['Hatchback', 'SUV', 'Sedan', 'MUV', 'Coupe', 'Minivans',
                                         'Convertibles', 'Hybrids', 'Wagon', 'Pickup Trucks'])
Tr = slt.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])
Owner = slt.sidebar.selectbox("Owner Count", [0, 1, 2, 3, 4, 5])
Brand = slt.sidebar.selectbox("Brand", options=df['Brand'].unique())

# Filter models based on selected Brand, Body Type, and Fuel Type
filtered_models = df[(df['Brand'] == Brand) & (df['body type'] == Bt) & (df['Fuel type'] == Ft)]['model'].unique()
Model = slt.sidebar.selectbox("Model", options=filtered_models)

Model_year = slt.sidebar.selectbox("Model Year", options=sorted(df['modelYear'].unique()))
IV = slt.sidebar.selectbox("Insurance Validity", ['Third Party insurance', 'Comprehensive', 'Third Party',
                                                  'Zero Dep', 'Not Available'])
Km = slt.sidebar.slider("Kilometers Driven", min_value=100, max_value=100000, step=1000)
ML = slt.sidebar.number_input("Mileage (kmpl)", min_value=5, max_value=50, step=1)
Seats = slt.sidebar.selectbox("Seats", options=sorted(df['Seats'].unique()))
Color = slt.sidebar.selectbox("Color", df['Color'].unique())
City = slt.sidebar.selectbox("City", options=df['City'].unique())

# Main layout for displaying selected data and predictions
col1, col2 = slt.columns(2)

# Display the selected car details
with col1:
    slt.subheader("📝 Selected Car Details")
    slt.markdown(
        f"""
        - **Fuel Type**: {Ft}
        - **Body Type**: {Bt}
        - **Transmission**: {Tr}
        - **Owner Count**: {Owner}
        - **Brand**: {Brand}
        - **Model**: {Model}
        - **Model Year**: {Model_year}
        - **Insurance Validity**: {IV}
        - **Kilometers Driven**: {Km}
        - **Mileage**: {ML}
        - **Seats**: {Seats}
        - **Color**: {Color}
        - **City**: {City}
        """
    )

# Prediction section
with col2:
    slt.subheader("💡 Price Prediction")
    slt.markdown("Click below to predict the resale value of the car:")

    if slt.button("Predict Car Price"):
        # Load the model pipeline
        with open('pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)

        # Prepare the input data for the model
        input_data = pd.DataFrame({
            'Fuel type': [Ft],
            'body type': [Bt],
            'transmission': [Tr],
            'ownerNo': [Owner],
            'Brand': [Brand],
            'model': [Model],
            'modelYear': [Model_year],
            'Insurance Validity': [IV],
            'Kms Driven': [Km],
            'Mileage': [ML],
            'Seats': [Seats],
            'Color': [Color],
            'City': [City]
        })

        # Predict the price using the pipeline
        prediction = pipeline.predict(input_data)

        # Display the prediction result
        slt.markdown(f"### 🏷️ Estimated Resale Value: **₹ {round(prediction[0], 2)}** lakhs")

# Footer with additional information
slt.write("---")
slt.markdown(
    """
    <div style='text-align: center;'>
        <p>🔧 Developed by <strong>Surenthiran S</strong> | Data Science Enthusiast 🚀</p>
    </div>
    """, 
    unsafe_allow_html=True
)
