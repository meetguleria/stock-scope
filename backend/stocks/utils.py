import os
import logging
import yfinance as yf
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_and_save_stock_data(ticker, start_date, end_date, interval='1d'):
    """Fetch and save stock data for a given ticker using yfinance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

        if data.empty:
            logging.warning(f"No data retrieved for {ticker} from {start_date} to {end_date} with interval {interval}")
            return None
        
        data = data.drop(columns=['Adj Close'], errors='ignore')
        file_path = os.path.join(DATA_DIR, f'{ticker}_daily.csv')
        data.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=True)
        
        logging.info(f"Data for {ticker} saved successfully to {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None
