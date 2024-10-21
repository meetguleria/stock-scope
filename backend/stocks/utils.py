import pandas as pd
import os
import logging
import joblib
from .feature_engineering import apply_feature_engineering_to_stock_data
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def load_latest_stock_data(ticker):
    """Load the most recent stock data for the given ticker from a CSV file."""
    file_path = os.path.join(DATA_DIR, f'{ticker}_daily.csv')
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' is a datetime object
        return df
    else:
        logging.warning(f"No CSV file found for {ticker} at {file_path}")
        return None

# Functions to load the models (as per your naming convention)
def load_gru_model(stock_symbol):
    model_path = os.path.join(os.path.dirname(__file__), '../models', f'gru_model_{stock_symbol}.pkl')
    return joblib.load(model_path)

def load_lstm_model(stock_symbol):
    model_path = os.path.join(os.path.dirname(__file__), '../models', f'lstm_model_{stock_symbol}.pkl')
    return joblib.load(model_path)

def load_random_forest_model(stock_symbol):
    model_path = os.path.join(os.path.dirname(__file__), '../models', f'rf_model_{stock_symbol}.pkl')
    return joblib.load(model_path)

def load_meta_model(stock_symbol):
    model_path = os.path.join(os.path.dirname(__file__), '../models', f'stacking_meta_model_{stock_symbol}.pkl')
    return joblib.load(model_path)

def load_scaler(stock_symbol, scaler_type):
    """Load the scaler for X or y."""
    scaler_path = os.path.join(os.path.dirname(__file__), '../models', f'minmax_scaler_{scaler_type}_{stock_symbol}.joblib')
    return joblib.load(scaler_path)

def fetch_and_save_stock_data(ticker, start_date, end_date, interval='1d'):
    """Fetch and save stock data for a given ticker."""
    try:
        # Fetch stock data using yfinance
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

        if data.empty:
            logging.warning(f"No data retrieved for {ticker} from {start_date} to {end_date} with interval {interval}")
            return None
        
        data = data.drop(columns=['Adj Close'], errors='ignore')

        # File path for the saved data
        file_path = os.path.join(DATA_DIR, f'{ticker}_daily.csv')

        # Save the new data
        if os.path.exists(file_path):
            data.to_csv(file_path, mode='a', header=False)
        else:
            data.to_csv(file_path, index=True)
        
        logging.info(f"Data for {ticker} saved successfully to {file_path}")
        return file_path
    
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None