import logging
from datetime import datetime
from celery import shared_task
from django.db import transaction
import pandas as pd
from .models import Stock, HistoricalData, Dividends, Financials, Sustainability
import yfinance as yf

# Configuring logging
logger = logging.getLogger(__name__)

@shared_task
def fetch_stock_data(symbols):
  for symbol in symbols:
    try:
      stock, created = Stock.objects.get_or_create(symbol=symbol)
      logger.info(f"{'Created new' if created else 'Fetched existing'} stock object for symbol: {symbol}")
      ticker = yf.Ticker(symbol)
      save_stock_data(stock, ticker)
    except Exception as e:
      logger.error(f"Failed to fetch or update data for {symbol}: {str(e)}")

def save_stock_data(stock, ticker):
  # Save historical stock data
  historical_data = ticker.history(period="1mo", interval="1d")
  save_historical_data(stock, historical_data)

  financial_data = {
    'income_statement': ticker.financials,
    'balance_sheet': ticker.balance_sheet,
    'cash_flow': ticker.cashflow
  }
  save_financial_data(stock, finacial_data)

def save_historical_data(stock, data):
  with transaction.atomic():
    for date, row in data.iterrows():
      HistoricalData.objects.update_or_create(
        stock=stock,
        date=date,
        defaults={
          'open_price': row['Open'],
          'high_price': row['High'],
          'low_price': row['Low'],
          'close_price': row['Close'],
          'volume': row['Volume'],
          'adjusted_close': row.get('Adj Close', row['Close'])
        }
      )

def save_financial_data(stock, data_frame, data_type):
  for data_type, data in financial_data.items():
    if data is not None and isinstance(data, pd.DataFrame):
      with transaction.atomic():
        for date, values in data.iterrows():
          defaults = {field.name: values.get(field.name) for field in Financials._meta.fields if field.name in values}
          defaults['data_type'] = data_type
          Financials.objects.update_or_create(stock=stock, date=date, defaults=defaults)
    else:
      logger.error(f"No data or incorrect for {data_type}")

def fetch_and_log_raw_data(ticker):
    try:
        data = ticker.history(period="1mo", interval="1d")
        if data.empty:
            logger.warning("No data fetched.")
        else:
            logger.debug(f"Raw data for {ticker.ticker}: {data.head()}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker.ticker}: {str(e)}")
        return pd.DataFrame()
