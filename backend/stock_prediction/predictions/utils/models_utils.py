from django.conf import settings
import os
import joblib
from tensorflow.keras.models import load_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_models():
    """
    Load all required models for prediction.
    
    Returns:
    dict: Dictionary of loaded models.
    """
    try:
        models = {
            'daily_orig': load_model(os.path.join(settings.MODELS_DIR, 'model_daily_orig.keras')),
            'daily_outlier': load_model(os.path.join(settings.MODELS_DIR, 'model_daily_outlier.keras')),
            'daily_ta': load_model(os.path.join(settings.MODELS_DIR, 'model_daily_ta.keras')),
            'hourly_orig': load_model(os.path.join(settings.MODELS_DIR, 'model_hourly_orig.keras')),
            'hourly_outlier': load_model(os.path.join(settings.MODELS_DIR, 'model_hourly_outlier.keras')),
            'hourly_ta': load_model(os.path.join(settings.MODELS_DIR, 'model_hourly_ta.keras')),
            'final': load_model(os.path.join(settings.MODELS_DIR, 'final_model_daily_with_hourly.keras'))
        }
        logging.info("Models loaded successfully.")
        return models
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        raise

def load_scaler(stock_symbol, data_type):
    """
    Load the scaler for the specified stock and data type.
    
    Parameters:
    stock_symbol (str): Stock symbol to load scaler for.
    data_type (str): Type of data ('daily' or 'hourly') to load the corresponding scaler.
    
    Returns:
    StandardScaler: Loaded scaler.
    """
    scaler_path = os.path.join(settings.MODELS_DIR, f'{stock_symbol}_{data_type}_scaler.pkl')
    try:
        if os.path.exists(scaler_path):
            logging.info(f"Scaler for {stock_symbol} ({data_type}) found at {scaler_path}.")
            return joblib.load(scaler_path)
        else:
            raise FileNotFoundError(f"Scaler file not found for stock: {stock_symbol} ({data_type})")
    except Exception as e:
        logging.error(f"Error loading scaler: {e}")
        raise
