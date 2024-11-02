import os
import joblib
import tensorflow as tf
import pickle
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models'))

MODEL_PATHS = {
    'scaler': lambda stock_symbol, scaler_type: os.path.join(MODEL_DIR, 'scalers', f'minmax_scaler_{scaler_type}_{stock_symbol.upper()}.joblib'),
    'gru': lambda stock_symbol: os.path.join(MODEL_DIR, 'gru_models', f'gru_{stock_symbol.upper()}_best.keras'),
    'lstm': lambda stock_symbol: os.path.join(MODEL_DIR, 'lstm_models', f'lstm_{stock_symbol.upper()}_best.keras'),
    'rf': lambda stock_symbol: os.path.join(MODEL_DIR, 'random_forest_models', f'rf_{stock_symbol.upper()}_model.joblib'),
    'xgb': lambda stock_symbol: os.path.join(MODEL_DIR, 'xgb_models', f'xgb_{stock_symbol.upper()}_model.json'),
    'meta': lambda stock_symbol: os.path.join(MODEL_DIR, 'meta_model', f'stacking_meta_model_{stock_symbol.upper()}.joblib'),
    'feature_names_rf': lambda stock_symbol: os.path.join(MODEL_DIR, 'random_forest_models', f'feature_names_{stock_symbol.upper()}.pkl'),
    'feature_names_xgb': lambda stock_symbol: os.path.join(MODEL_DIR, 'xgb_models', f'feature_names_{stock_symbol.upper()}.pkl'),
    'meta_feature_names': lambda stock_symbol: os.path.join(MODEL_DIR, 'meta_model', f'meta_feature_names_{stock_symbol.upper()}.pkl')
}

# Load Scaler
def load_scaler(stock_symbol, scaler_type='X'):
    scaler_path = MODEL_PATHS['scaler'](stock_symbol, scaler_type)
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler file not found: {scaler_path}")
        return None
    try:
        scaler = joblib.load(scaler_path)
        logger.info(f"Scaler '{scaler_type}' loaded for {stock_symbol}")
        return scaler
    except Exception as e:
        logger.exception(f"Error loading scaler '{scaler_type}' for {stock_symbol}: {e}")
        raise

# Load GRU model
def load_gru_model(stock_symbol):
    model_path = MODEL_PATHS['gru'](stock_symbol)
    logger.info(f"Attempting to load GRU model from path: {model_path}")
    return _load_keras_model(stock_symbol, model_path, "GRU")

# Load LSTM model
def load_lstm_model(stock_symbol):
    model_path = MODEL_PATHS['lstm'](stock_symbol)
    logger.info(f"Attempting to load LSTM model from path: {model_path}")
    return _load_keras_model(stock_symbol, model_path, "LSTM")

def _load_keras_model(stock_symbol, model_path, model_type):
    if not os.path.exists(model_path):
        logger.error(f"{model_type} model file not found: {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"{model_type} model loaded for {stock_symbol}")
        return model
    except Exception as e:
        logger.exception(f"Error loading {model_type} model for {stock_symbol} from {model_path}: {e}")
        raise

# Load Random Forest model
def load_random_forest_model(stock_symbol):
    model_path = MODEL_PATHS['rf'](stock_symbol)
    if not os.path.exists(model_path):
        logger.error(f"Random Forest model file not found: {model_path}")
        return None
    try:
        rf_model = joblib.load(model_path)
        logger.info(f"Random Forest model loaded for {stock_symbol}")
        return rf_model
    except Exception as e:
        logger.exception(f"Error loading Random Forest model for {stock_symbol}: {e}")
        raise

# Load XGBoost model
def load_xgboost_model(stock_symbol):
    model_path = MODEL_PATHS['xgb'](stock_symbol)
    if not os.path.exists(model_path):
        logger.error(f"XGBoost model file not found: {model_path}")
        return None
    try:
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(model_path)
        logger.info(f"XGBoost model loaded for {stock_symbol}")
        return xgb_model
    except Exception as e:
        logger.exception(f"Error loading XGBoost model for {stock_symbol}: {e}")
        raise

# Load Feature Names
def load_feature_names(stock_symbol, model_type):
    if model_type == 'random_forest':
        feature_names_path = MODEL_PATHS['feature_names_rf'](stock_symbol)
    elif model_type == 'xgboost':
        feature_names_path = MODEL_PATHS['feature_names_xgb'](stock_symbol)
    else:
        logger.error(f"Invalid model type specified: {model_type}")
        return None

    if not os.path.exists(feature_names_path):
        logger.error(f"Feature names file not found: {feature_names_path}")
        return None
    try:
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)
        logger.info(f"Feature names loaded for {stock_symbol} ({model_type})")
        return feature_names
    except Exception as e:
        logger.exception(f"Failed to load feature names for {stock_symbol} ({model_type}): {e}")
        raise

# Load Meta-Feature Names
def load_meta_feature_names(stock_symbol):
    feature_names_path = MODEL_PATHS['meta_feature_names'](stock_symbol)
    if not os.path.exists(feature_names_path):
        logger.error(f"Meta-feature names file not found: {feature_names_path}")
        return None
    try:
        with open(feature_names_path, 'rb') as f:
            meta_feature_names = pickle.load(f)
        logger.info(f"Meta-feature names loaded for {stock_symbol}")
        return meta_feature_names
    except Exception as e:
        logger.exception(f"Failed to load meta-feature names for {stock_symbol}: {e}")
        raise

# Load Meta Model
def load_meta_model(stock_symbol):
    model_path = MODEL_PATHS['meta'](stock_symbol)
    if not os.path.exists(model_path):
        logger.error(f"Meta-model file not found: {model_path}")
        return None

    try:
        meta_model = joblib.load(model_path)
        logger.info(f"Meta-model loaded successfully for {stock_symbol}")
        return meta_model
    except Exception as e:
        logger.exception(f"Error loading Meta-model for {stock_symbol}: {e}")
        raise
