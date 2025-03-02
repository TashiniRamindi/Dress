import streamlit as st
import joblib
import pandas as pd
import numpy as np

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
    'Collar': st.selectbox('Collar', ['yes', 'no']),
    'Neckline': st.selectbox('Neckline', ['round', 'v_neck', 'square', 'collared']),
    'Hemline': st.selectbox('Hemline', ['straight', 'curved']),
    'Style': st.selectbox('Style', ['casual', 'formal']),
    'Sleeve Style': st.selectbox('Sleeve Style', ['short', 'long']),
    'Pattern': st.selectbox('Pattern', ['solid', 'striped', 'floral']),
    'Product Colour': st.selectbox('Product Colour', ['red', 'blue', 'green', 'black', 'white']),
    'Material': st.selectbox('Material', ['cotton', 'polyester', 'wool', 'silk'])
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

