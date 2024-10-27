import yfinance as yf
import pandas as pd
import os
import logging
from datetime import datetime

# Directory to store stock data
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Fetches stock data for a specified range and saves it to a CSV
def fetch_and_save_stock_data(ticker, start_date, end_date, interval='1d'):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if data.empty:
            logging.warning(f"No data retrieved for {ticker}")
            return None
        data = data.drop(columns=['Adj Close'], errors='ignore')
        file_path = os.path.join(DATA_DIR, f'{ticker}_daily.csv')
        data.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=True)
        logging.info(f"Data for {ticker} saved to {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None

def load_latest_stock_data(ticker):
    """Load the most recent stock data for the given ticker."""
    file_path = os.path.join(DATA_DIR, f'{ticker}_daily.csv')
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    else:
        logging.warning(f"No CSV file found for {ticker} at {file_path}")
        # Attempt to fetch data if the file is missing
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now().replace(year=datetime.now().year - 1)).strftime('%Y-%m-%d')
        fetch_and_save_stock_data(ticker, start_date, end_date)
        
        # Reload the data after fetching
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        return None
