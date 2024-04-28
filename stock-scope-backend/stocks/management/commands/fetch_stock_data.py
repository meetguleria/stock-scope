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
    
    # Check if DataFrame has MultiIndex Columns and adjust accordingly
    if isinstance(data.columns, pd.MultiIndex):
      data.columns = data.columns.map('_'.join)

    for symbol in symbols:
      stock, _ = Stock.objects.get_or_create(symbol=symbol)
      for date, row in data.iterrows():
          defaults = {
            'open_price': row.get(f'Open_{symbol}', 0),
            'high_price': row.get(f'High_{symbol}', 0),
            'low_price': row.get(f'Low_{symbol}', 0),
            'close_price': row.get(f'Close_{symbol}', 0),
            'volume': row.get(f'Volume_{symbol}', 0),
            'adjusted_close': row.get(f'Adj Close_{symbol}', None)
          }

          print(f"Updating {symbol} on {date} with: {defaults}")
          
          try:
            HistoricalData.objects.update_or_create(
              stock=stock,
              date=date,
              defaults=defaults
            )
          except IntegrityError as e:
            self.stdout.write(self.style.ERROR(f'Failed to update database for {symbol} on {date}: {str(e)}'))
            print(f"Failed to update {symbol} on {date} with: {defaults}")
            
    self.stdout.write(self.style.SUCCESS('Successfully fetched and updated stock data'))
