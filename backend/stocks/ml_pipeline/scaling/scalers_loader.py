import os
import joblib
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

# Path to saved scalers directory
SCALER_DIR = os.path.join(settings.BASE_DIR, 'models', 'scalers')

def load_scalers(stock_symbol):
    try:
        scaler_X_path = os.path.join(SCALER_DIR, f'minmax_scaler_X_{stock_symbol}.joblib')
        scaler_y_path = os.path.join(SCALER_DIR, f'minmax_scaler_y_{stock_symbol}.joblib')

        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)

        logger.info(f"Scalers loaded successfully for {stock_symbol}")
        return scaler_X, scaler_y
    except Exception as e:
        logger.error(f"Failed to load scalers for {stock_symbol}: {e}")
        raise e

def scale_features(stock_symbol, features):
    scaler_X, _ = load_scalers(stock_symbol)
    try:
        scaled_features = scaler_X.transform([features])
        logger.info(f"Features scaled successfully for {stock_symbol}")
        return scaled_features
    except Exception as e:
        logger.error(f"Error scaling features for {stock_symbol}: {e}")
        raise e
