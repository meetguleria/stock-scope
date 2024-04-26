import pandas as pd
from django.core.management.base import BaseCommand
import yfinance as yf
from stocks.models import Stock, HistoricalData

class Command(BaseCommand):
  help = 'Fetches stock data from Yahoo Finance'

  def add_arguments(self, parser):
    parser.add_argument('-s', '--symbols', type=str, help='Stock symbols comma-separated (e.g., "AAPL,MSFT,GOOGL")')

  def handle(self, *args, **options):
    symbols = options['symbols'].split(',') if options['symbols'] else ['AAPL', 'MSFT', 'GOOGL']
    data = yf.download(symbols, period="1mo", interval="1d")

    print(data.head())
    print(data.columns)
    
    if isinstance(data.columns, pd.MultiIndex):
      data.columns = data.columns.map('_'.join)  # Flattens MultiIndex to single index

    for symbol in symbols:
      stock, _ = Stock.objects.get_or_create(symbol=symbol)
      for date, row in data[symbol].iterrows():
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
    self.stdout.write(self.style.SUCCESS('Successfully fetched and updated stock data'))
