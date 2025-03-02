import streamlit as st
import joblib
import pandas as pd
import numpy as np
import base64

# Function to handle image encoding
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

# Function to set an image at the top (instead of as background)
def display_image(image_path):
    base64_str = get_base64_image(image_path)
    st.markdown(f"""
    <div style="text-align: center;">
        <img src="data:image/jpeg;base64,{base64_str}" width="400" height="300">
    </div>
    """, unsafe_allow_html=True)

display_image("background.jpg")  

# Load the saved model and columns
model = joblib.load("classification_model_dress.pkl")
columns = joblib.load("dress_X_train.pkl")

# Function to preprocess inputs based on user data
def preprocess_input(user_input):
    # One-Hot Encoding for categorical columns
    dummy_cols = ['Collar', 'Neckline', 'Hemline', 'Style', 'Sleeve Style', 'Pattern', 'Product Colour', 'Material']
    input_df = pd.DataFrame([user_input], columns=user_input.keys())
    
    input_dummies = pd.get_dummies(input_df[dummy_cols], drop_first=True)
    input_df = pd.concat([input_df, input_dummies], axis=1)
    input_df = input_df.drop(columns=dummy_cols)
    
    # Ordinal Encoding for specific columns
    fit_mapping = {'slim_fit': 0, 'regular_fit': 1, 'relaxed_fit': 3}
    length_mapping = {'mini': 0, 'knee': 1, 'midi': 2, 'maxi': 3}
    sleeve_length_mapping = {'sleeveless': 0, 'short_length': 1, 'elbow_length': 2, 'three_quarter_sleeve': 3, 'long_sleeve': 4}
    
    input_df['Fit'] = input_df['Fit'].map(fit_mapping)
    input_df['Length'] = input_df['Length'].map(length_mapping)
    input_df['Sleeve Length'] = input_df['Sleeve Length'].map(sleeve_length_mapping)
    
    # Add the new features from radio buttons (Yes=1, No=0)
    input_df['Breathable'] = 1 if user_input['Breathable'] == 'Yes' else 0
    input_df['Lightweight'] = 1 if user_input['Lightweight'] == 'Yes' else 0
    input_df['Water_Repellent'] = 1 if user_input['Water_Repellent'] == 'Yes' else 0
    
    # Reindex to match the columns the model was trained on
    input_df = input_df.reindex(columns=columns, fill_value=0)
    
    return input_df

# Streamlit app interface
st.title("Dress Season Prediction App")
st.write("Provide the details of the dress to predict the season.")

# Initialize user input to None
user_input = {}

# Add user inputs for dress features
fit = st.selectbox('Fit', ['', 'slim_fit', 'regular_fit', 'relaxed_fit'])
length = st.selectbox('Length', ['', 'mini', 'knee', 'midi', 'maxi'])
sleeve_length = st.selectbox('Sleeve Length', ['', 'sleeveless', 'short_length', 'elbow_length', 'three_quarter_sleeve', 'long_sleeve'])
collar = st.selectbox('Collar', ['', 'shirt_collar', 'Basic', 'other_collar', 'no_collar', 'high_collar', 'polo_collar', 'Ruffled/Decorative'])
neckline = st.selectbox('Neckline', ['', 'other_neckline', 'collared_neck', 'off_shoulder', 'v_neck', 'high_neck', 'sweetheart_neck', 'crew_neck', 'square_neck'])
hemline = st.selectbox('Hemline', ['', 'curved_hem', 'straight_hem', 'other_hemline', 'asymmetrical_hem', 'flared_hem', 'ruffle_hem'])
style = st.selectbox('Style', ['', 'fit_and_flare', 'sundress', 'sweater & jersey', 'other_style', 'shirtdress & tshirt', 'babydoll', 'slip', 'a_line'])
sleeve_style = st.selectbox('Sleeve Style', ['', 'ruched', 'cuff', 'ruffle', 'bishop_sleeve', 'plain', 'other_sleeve_style', 'balloon', 'puff', 'kimono', 'no_sleeve', 'cap'])
pattern = st.selectbox('Pattern', ['', 'floral_prints', 'animal_prints', 'other', 'multicolor', 'cable_knit', 'printed', 'other_pattern', 'stripes_and_checks', 'solid_or_plain', 'polka_dot'])
product_colour = st.selectbox('Product Colour', ['', 'green', 'grey', 'pink', 'brown', 'metallics', 'blue', 'neutral', 'white', 'black', 'orange', 'purple', 'multi_color', 'red', 'yellow'])
material = st.selectbox('Material', ['', 'Other', 'Synthetic Fibers', 'Wool', 'Silk', 'Luxury Materials', 'Cotton', 'Metallic', 'Knitted and Jersey Materials', 'Leather', 'Polyester'])

# Radio buttons for additional features (no default selection)
breathable = st.radio('Is the dress breathable?', ['Yes', 'No'], index=-1)
lightweight = st.radio('Is the dress lightweight?', ['Yes', 'No'], index=-1)
water_repellent = st.radio('Is the dress water repellent?', ['Yes', 'No'], index=-1)


# Check if user clicked predict or cancel button
if st.button('Predict Season'):
    # Ensure all fields are filled before processing
    if fit and length and sleeve_length and collar and neckline and hemline and style and sleeve_style and pattern and product_colour and material and breathable and lightweight and water_repellent:
        # Store inputs in user_input dictionary
        user_input = {
            'Fit': fit,
            'Length': length,
            'Sleeve Length': sleeve_length,
            'Collar': collar,
            'Neckline': neckline,
            'Hemline': hemline,
            'Style': style,
            'Sleeve Style': sleeve_style,
            'Pattern': pattern,
            'Product Colour': product_colour,
            'Material': material,
            'Breathable': breathable,
            'Lightweight': lightweight,
            'Water_Repellent': water_repellent
        }

        # Preprocess the input data
        processed_input = preprocess_input(user_input)
        
        # Predict the season
        prediction = model.predict(processed_input)[0]
        
        # Convert the predicted label to the actual season
        season_mapping = {0: 'spring', 1: 'summer', 2: 'winter', 3: 'autumn'}
        predicted_season = season_mapping[prediction]
        
        # Display the prediction
        st.write(f"The predicted season for the given dress is: **{predicted_season.capitalize()}**")
    else:
        st.write("Please fill in all the fields before predicting.")
elif st.button('Cancel'):
    # Clear user inputs (reset to initial state)
    st.experimental_rerun()



