import os
import joblib
import tensorflow as tf
import pickle
import xgboost as xgb

# Adjust path to point to models in the backend root folder
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def load_scaler(stock_symbol, scaler_type='X'):
    scaler_path = os.path.join(MODEL_DIR, 'scalers', f'minmax_scaler_{scaler_type}_{stock_symbol}.joblib')
    return joblib.load(scaler_path)

def load_gru_model(stock_symbol):
    model_path = os.path.join(MODEL_DIR, 'gru_models', f'gru_{stock_symbol}_best.keras')
    return tf.keras.models.load_model(model_path)

def load_lstm_model(stock_symbol):
    model_path = os.path.join(MODEL_DIR, 'lstm_models', f'lstm_{stock_symbol}_best.keras')
    return tf.keras.models.load_model(model_path)

def load_random_forest_model(stock_symbol):
    model_path = os.path.join(MODEL_DIR, 'random_forest_models', f'rf_{stock_symbol}_model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def load_xgb_model(stock_symbol):
    model_path = os.path.join(MODEL_DIR, 'xgb_models', f'xgb_{stock_symbol}_model.json')
    return xgb.Booster(model_file=model_path)

def load_meta_model(stock_symbol):
    model_path = os.path.join(MODEL_DIR, 'meta_model', f'stacking_meta_model_{stock_symbol}.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)
