import streamlit as st
import joblib
import pandas as pd
import numpy as np
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

def set_background(image_path):
    base64_str = get_base64_image(image_path)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background("background.jpg")

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

# User inputs for dress features
user_input = {
    'Fit': st.selectbox('Fit', ['slim_fit', 'regular_fit', 'relaxed_fit']),
    'Length': st.selectbox('Length', ['mini', 'knee', 'midi', 'maxi']),
    'Sleeve Length': st.selectbox('Sleeve Length', ['sleeveless', 'short_length', 'elbow_length', 'three_quarter_sleeve', 'long_sleeve']),
    'Collar': st.selectbox('Collar', ['shirt_collar', 'Basic', 'other_collar', 'no_collar', 'high_collar', 'polo_collar', 'Ruffled/Decorative']),
    'Neckline': st.selectbox('Neckline', ['other_neckline', 'collared_neck', 'off_shoulder', 'v_neck', 'high_neck', 'sweetheart_neck', 'crew_neck', 'square_neck']),
    'Hemline': st.selectbox('Hemline', ['curved_hem', 'straight_hem', 'other_hemline', 'asymmetrical_hem', 'flared_hem', 'ruffle_hem']),
    'Style': st.selectbox('Style', ['fit_and_flare', 'sundress', 'sweater & jersey', 'other_style', 'shirtdress & tshirt', 'babydoll', 'slip', 'a_line']),
    'Sleeve Style': st.selectbox('Sleeve Style', ['ruched', 'cuff', 'ruffle', 'bishop_sleeve', 'plain', 'other_sleeve_style', 'balloon', 'puff', 'kimono', 'no_sleeve', 'cap']),
    'Pattern': st.selectbox('Pattern', ['floral_prints', 'animal_prints', 'other', 'multicolor', 'cable_knit', 'printed', 'other_pattern', 'stripes_and_checks', 'solid_or_plain', 'polka_dot']),
    'Product Colour': st.selectbox('Product Colour', ['green', 'grey', 'pink', 'brown', 'metallics', 'blue', 'neutral', 'white', 'black', 'orange', 'purple', 'multi_color', 'red', 'yellow']),
    'Material': st.selectbox('Material', ['Other', 'Synthetic Fibers', 'Wool', 'Silk', 'Luxury Materials', 'Cotton', 'Metallic', 'Knitted and Jersey Materials', 'Leather', 'Polyester']),
    
    # New radio buttons for additional features (Yes/No)
    'Breathable': st.radio('Is the dress breathable?', ['Yes', 'No']),
    'Lightweight': st.radio('Is the dress lightweight?', ['Yes', 'No']),
    'Water_Repellent': st.radio('Is the dress water repellent?', ['Yes', 'No'])
}

# When user clicks the button
if st.button('Predict Season'):
    # Preprocess the input data
    processed_input = preprocess_input(user_input)
    
    # Predict the season
    prediction = model.predict(processed_input)[0]
    
    # Convert the predicted label to the actual season
    season_mapping = {0: 'spring', 1: 'summer', 2: 'winter', 3: 'autumn'}
    predicted_season = season_mapping[prediction]
    
    # Display the prediction
    st.write(f"The predicted season for the given dress is: **{predicted_season.capitalize()}**")


