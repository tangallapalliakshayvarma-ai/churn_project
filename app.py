# Integrating ANN model with Streamlit web app
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import tensorflow as tf
import pickle

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# --- Load the trained model ---
model = tf.keras.models.load_model('ann_model.h5')

# --- Load encoders and scaler ---
with open('label_encoder_gender.pk1', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pk1', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pk1', 'rb') as file:
    scaler = pickle.load(file)

st.title('📊 Customer Churn Prediction')

# --- User input widgets ---
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92, value=30)
balance = st.number_input('Balance', min_value=0.0, value=0.0)
credit_score = st.number_input('Credit Score', min_value=0, value=600)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
tenure = st.slider('Tenure (years with bank)', 0, 10, value=3)
num_of_products = st.slider('Number of Products', 1, 4, value=1)
has_cr_card = st.selectbox('Has Credit Card?', [0, 1])
is_active_member = st.selectbox('Is Active Member?', [0, 1])

# --- Prepare input data frame (numeric + encoded) ---
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

try:
    geo_col_names = onehot_encoder_geo.get_feature_names_out(['Geography'])
except AttributeError:
    # Older sklearn fallback
    categories = onehot_encoder_geo.categories_[0]
    geo_col_names = [f'Geography_{cat}' for cat in categories]

geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_col_names)

# Combine with numeric features
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# --- Align columns with scaler / training ---
if hasattr(scaler, 'feature_names_in_'):
    feature_cols = list(scaler.feature_names_in_)
    input_data = input_data.reindex(columns=feature_cols, fill_value=0)
else:
    # Fallback if scaler has no feature names (trained on NumPy)
    # Make sure this matches the order used during training
    feature_cols = [
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'Geography_France', 'Geography_Germany', 'Geography_Spain'
    ]
    input_data = input_data.reindex(columns=feature_cols, fill_value=0)

# --- Scale the input ---
try:
    input_data_scaled = scaler.transform(input_data)
except ValueError:
    # If scaler complains about feature names, transform values only
    input_data_scaled = scaler.transform(input_data.values)

# --- Predict ---
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    prediction_proba = float(prediction[0][0])

    st.write(f'**Churn Probability:** `{prediction_proba:.4f}`')

    if prediction_proba > 0.5:
        st.error('🔴 The customer is **likely to churn**.')
    else:
        st.success('🟢 The customer is **not likely to churn**.')
