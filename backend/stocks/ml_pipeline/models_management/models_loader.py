import os
import joblib
import tensorflow as tf
import pickle5 as pickle
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

MODEL_PATHS = {
    'scaler': lambda stock_symbol, scaler_type: os.path.join(MODEL_DIR, 'scalers', f'minmax_scaler_{scaler_type}_{stock_symbol}.joblib'),
    'gru': lambda stock_symbol: os.path.join(MODEL_DIR, 'gru_models', f'gru_{stock_symbol}_best.keras'),
    'lstm': lambda stock_symbol: os.path.join(MODEL_DIR, 'lstm_models', f'lstm_{stock_symbol}_best.keras'),
    'rf': lambda stock_symbol: os.path.join(MODEL_DIR, 'random_forest_models', f'rf_{stock_symbol}_model.pkl'),
    'xgb': lambda stock_symbol: os.path.join(MODEL_DIR, 'xgb_models', f'xgb_{stock_symbol}_model.json'),
    'meta': lambda stock_symbol: os.path.join(MODEL_DIR, 'meta_model', f'stacking_meta_model_{stock_symbol}.pkl')
}

# Load Scaler
def load_scaler(stock_symbol, scaler_type='X'):
    scaler_path = MODEL_PATHS['scaler'](stock_symbol, scaler_type)
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler file not found: {scaler_path}")
        return None
    try:
        return joblib.load(scaler_path)
    except Exception as e:
        logger.error(f"Error loading scaler '{scaler_type}' for {stock_symbol}: {e}")
        raise e

# Load GRU model
def load_gru_model(stock_symbol):
    model_path = MODEL_PATHS['gru'](stock_symbol)
    logger.info(f"Attempting to load GRU model from path: {model_path}")
    return _load_keras_model(stock_symbol, model_path, "GRU")

# Load LSTM model
def load_lstm_model(stock_symbol):
    return _load_keras_model(stock_symbol, MODEL_PATHS['lstm'](stock_symbol), "LSTM")

def _load_keras_model(stock_symbol, model_path, model_type):
    if not os.path.exists(model_path):
        logger.error(f"{model_type} model file not found: {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"{model_type} model loaded for {stock_symbol}")
        return model
    except Exception as e:
        logger.error(f"Error loading {model_type} model for {stock_symbol} from {model_path}: {e}")
        raise e

# Load Random Forest model
def load_random_forest_model(stock_symbol):
    model_path = MODEL_PATHS['rf'](stock_symbol)
    if not os.path.exists(model_path):
        logger.error(f"Random Forest model file not found: {model_path}")
        return None
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading Random Forest model for {stock_symbol}: {e}")
        raise e

# Load XGBoost model
def load_xgb_model(stock_symbol):
    model_path = MODEL_PATHS['xgb'](stock_symbol)
    if not os.path.exists(model_path):
        logger.error(f"XGBoost model file not found: {model_path}")
        return None
    try:
        model = xgb.Booster(model_file=model_path)
        logger.info(f"XGBoost model loaded for {stock_symbol}")
        return model
    except Exception as e:
        logger.error(f"Error loading XGBoost model for {stock_symbol}: {e}")
        raise e

# Load Meta-model
def load_meta_model(stock_symbol):
    model_path = MODEL_PATHS['meta'](stock_symbol)
    if not os.path.exists(model_path):
        logger.error(f"Meta-model file not found: {model_path}")
        return None
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Meta-model loaded for {stock_symbol}")
        return model
    except pickle.UnpicklingError as pe:
        logger.error(f"UnpicklingError loading Meta-model for {stock_symbol}: {pe}")
        raise pe
    except Exception as e:
        logger.error(f"Error loading Meta-model for {stock_symbol}: {e}")
        raise e
