import logging
from datetime import datetime
from celery import shared_task
from django.db import transaction
import pandas as pd
from .models import Stock, HistoricalData, Dividends, Financials, Sustainability
import yfinance as yf

# Configuring logging
logger = logging.getLogger(__name__)

def is_unix_timestamp(value):
    """Check if the given value is a Unix timestamp."""
    try:
      datetime.utcfromtimestamp(value)
      return True
    except (ValueError, OverflowError):
      return False


def convert_unix_to_isoformat(timestamp):
  """Convert Unix timestamp to ISO format date string."""
  return datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')

@shared_task
def fetch_stock_data(symbols):
  for symbol in symbols:
    try:
      stock, created = Stock.objects.get_or_create(symbol=symbol)
      logger.info(f"{'Created new' if created else 'Fetched existing'} stock object for symbol: {symbol}")
      ticker = yf.Ticker(symbol)

      data = ticker.history(period="1mo", interval="1d")
      if data.empty:
        logger.warning(f"No historical data fetched for symbol: {symbol}")
        continue

      logger.debug(f"Data fetched for {symbol}: {data.head()}")
      process_historical_data(stock, data)
      process_financial_data(stock, ticker)

    except Exception as e:
      logger.error(f"Failed to fetch or update data for {symbol}: {str(e)}")

def process_historical_data(stock, data):
  with transaction.atomic():
    for date, row in data.iterrows():
      date_str = date.strftime('%Y-%m-%d')
      HistoricalData.objects.update_or_create(
        stock=stock,
        date=date_str,
        defaults={
          'open_price': row['Open'],
          'high_price': row['High'],
          'low_price': row['Low'],
          'close_price': row['Close'],
          'volume': row['Volume'],
          'adjusted_close': row.get('Adj Close', row['Close'])
        }
      )
      logger.debug(f"Saved historical data for {stock.symbol} on {date_str}")


def process_financial_data(stock, ticker):
  with transaction.atomic():
    # Extract income statement data
    income_data = ticker.financials
    save_financial_data(stock, income_data, 'income_stmt')

    # Extract balance sheet data
    balance_sheet_data = ticker.balance_sheet
    save_financial_data(stock, balance_sheet_data, 'balance_sheet')

    # Extract cash flow statement data
    cashflow_data = ticker.cashflow
    save_financial_data(stock, cashflow_data, 'cashflow')

def transform_dataframe(df):
  original_index = df.index.copy()
  try:
    df.index = pd.to_datetime(df.index, errors='coerce')
    if df.index.hasnans:
      logger.error(f"Conversion resulted in NaNs, original index values: {original_index}")
      return None
    logger.debug("Converted index to DateTimeIndex successfully.")
  except Exception as e:
    logger.error(f"Failed to convert index to DatetimeIndex: {str(e)}, original index values: {original_index}")
    df = None
  return df

def save_financial_data(stock, data_frame, data_type):
  """Save only relevant fields to Financials model."""
  relevant_fields = {
    'income_stmt': {
      'net_income': 'Net Income',
      'total_revenue': 'Total Revenue',
      'ebit': 'EBIT',
      'operating_income': 'Operating Income',
      'earnings_before_tax': 'Pretax Income',
      'operating_cashflow': 'Operating Cash Flow'
    },
    'balance_sheet': {
      'total_assets': 'Total Assets',
      'total_liabilities': 'Total Liabilities',
      'total_equity': 'Stockholders Equity',
      'net_tangible_assets': 'Net Tangible Assets'
    },
    'cashflow': {
      'free_cashflow': 'Free Cash Flow',
      'capital_expenditure': 'Capital Expenditure',
      'net_income': 'Net Income From Continuing Operations',
      'operating_cashflow': 'Operating Cash Flow'
    }
  }

  if data_type not in relevant_fields:
    logger.warning(f"Data type {data_type} not recognized.")
    return
  
  data_frame = transform_dataframe(data_frame)
  if data_frame is None:
    logger.warning(f"Skipping saving financial data for {stock.symbol} due to index conversion failure.")
    return

  for date, values in data_frame.iterrows():
    date_str = data.strftime('%Y-%m-%d')
    defaults = {field: values.get(source_field, 0) for field, source_field in relevant_fields[data_type].items()}
    defaults['data_type'] = data_type
    Financials.objects.update_or_create(
      stock=stock, date=date_str, defaults=defaults
    )
    logger.debug(f"Saved financial data for {stock.symbol} under")

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
