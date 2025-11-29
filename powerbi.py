import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import os

# --- Load the Data, Model, and Preprocessing Objects ---
try:
    # 1. Load the original dataset
    df = pd.read_csv('Churn_Modelling.csv')

    # 2. Load the trained model and preprocessing objects
    model = load_model('ann_model.h5')

    with open('scaler.pk1', 'rb') as f:
        scaler = pickle.load(f)

    with open('label_encoder_gender.pk1', 'rb') as f:
        label_encoder_gender = pickle.load(f)

    with open('onehot_encoder_geo.pk1', 'rb') as f:
        onehot_encoder_geo = pickle.load(f)

    print("✅ All resources loaded successfully.")

except FileNotFoundError as e:
    print("\n❌ FATAL ERROR: A required file was not found.")
    print(f"   File not found: {e}")
    raise SystemExit

# --- 3. Prepare Features for Prediction (must match training data) ---

# Drop non-predictor columns
X = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited']).copy()
y_true = df['Exited']

# A. Encode Gender (LabelEncoder)
X['Gender'] = label_encoder_gender.transform(X['Gender'])

# B. OneHotEncode Geography
geo_features = X[['Geography']].values
geo_encoded = onehot_encoder_geo.transform(geo_features).toarray()

# Geography column names (same as during training)
try:
    geo_col_names = onehot_encoder_geo.get_feature_names_out(['Geography'])
except AttributeError:
    # Older sklearn fallback (if needed)
    categories = onehot_encoder_geo.categories_[0]
    geo_col_names = [f'Geography_{cat}' for cat in categories]

geo_df = pd.DataFrame(geo_encoded, columns=geo_col_names)

# Drop original Geography
X = X.drop('Geography', axis=1)

# Reset index for safe concat
X.reset_index(drop=True, inplace=True)
geo_df.reset_index(drop=True, inplace=True)

# C. Concatenate all features
X_processed = pd.concat([X, geo_df], axis=1)

# --- Ensure column order matches scaler ---

if hasattr(scaler, 'feature_names_in_'):
    feature_cols = list(scaler.feature_names_in_)
    print("Using feature order from scaler.feature_names_in_:")
    print(feature_cols)
    X_processed = X_processed.reindex(columns=feature_cols, fill_value=0)
else:
    # Fallback: manually specify (must match training order!)
    feature_cols = [
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'Geography_France', 'Geography_Germany', 'Geography_Spain'
    ]
    print("Scaler has no feature names; using manual feature_cols.")
    X_processed = X_processed.reindex(columns=feature_cols, fill_value=0)

# D. Scale
try:
    X_scaled = scaler.transform(X_processed)
except ValueError:
    # In case scaler was fit without feature names and complains
    X_scaled = scaler.transform(X_processed.values)

print("✅ Data successfully processed and scaled.")

# --- 4. Generate Predictions and Combine Data ---

churn_probabilities = model.predict(X_scaled)

df['Churn_Probability'] = churn_probabilities.flatten()
df = df.rename(columns={'Exited': 'Actual_Churn_Status'})

df['Predicted_Churn_Class'] = (df['Churn_Probability'] > 0.5).astype(int)

# Final export for Power BI
power_bi_data = df[[
    'CustomerId',
    'Geography',
    'Gender',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'EstimatedSalary',
    'Actual_Churn_Status',
    'Churn_Probability',
    'Predicted_Churn_Class'
]]

output_file = 'ann_churn_predictions_for_powerbi.csv'
power_bi_data.to_csv(output_file, index=False)

print(f"\n✅ Export complete! File created: {output_file}")
