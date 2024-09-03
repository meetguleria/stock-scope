import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Directory to save the scaler files
model_save_dir = "../models"
os.makedirs(model_save_dir, exist_ok=True)

def scale_dataframe(df, scaler_name, save_scaler=True):

    scaler_path = os.path.join(model_save_dir, f'{scaler_name}_scaler.pkl')
    
    # Check if the scaler already exists (if we want to reuse it)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()

    # Select numerical columns for scaling
    features_to_scale = df.select_dtypes(include=[np.number])
    
    # Exclude 'OutlierFlag' if it exists
    if 'OutlierFlag' in df.columns:
        features_to_scale = features_to_scale.drop(columns=['OutlierFlag'], errors='ignore')
    
    # If there are no numerical features left after dropping 'OutlierFlag', return the original DataFrame
    if features_to_scale.empty:
        print(f"No numerical features to scale for {scaler_name}.")
        return df, scaler
    
    # Scale the features
    scaled_features = pd.DataFrame(scaler.fit_transform(features_to_scale), columns=features_to_scale.columns, index=df.index)
    
    # Save the scaler in the models directory, if required
    if save_scaler and not os.path.exists(scaler_path):
        joblib.dump(scaler, scaler_path)
    
    # Combine scaled features with 'OutlierFlag' and other non-numerical columns if they exist
    non_scaled_columns = df.drop(columns=features_to_scale.columns, errors='ignore')
    scaled_df = pd.concat([scaled_features, non_scaled_columns], axis=1)

    return scaled_df, scaler

def inverse_transform(scaled_data, scaler_name):

    scaler_path = os.path.join(model_save_dir, f'{scaler_name}_scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler {scaler_name} not found.")
    
    scaler = joblib.load(scaler_path)
    return scaler.inverse_transform(scaled_data)
