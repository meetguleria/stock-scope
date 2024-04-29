from celery import shared_task
from django.db import transaction
from .models import Stock, HistoricalData
import yfinance as yf

@shared_task
def fetch_stock_data(symbol):
  for symbol in symbols:
    try:
      stock = Stock.objects.get(symbol=symbol)
      data = yf.download(symbol, period="1mo", interval="1d")
      if data.empty:
        raise ValueError(f"No data fetched for symbol: {symbol}")

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
              'volume': row['Volume']
            }
          )
    except Exception as e:
      print(f"Failed to fetch or update data for {symbol}: {str(e)}")
