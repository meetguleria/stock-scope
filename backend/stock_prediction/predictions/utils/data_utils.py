import yfinance as yf

def fetch_ohlc_data(stock, period=None, interval='1d'):
    # Set default periods based on the interval to ensure sufficient data
    if interval == '1d' and not period:
        period = '6mo'  # 6 months of daily data
    elif interval == '1h' and not period:
        period = '90d'  # 90 days of hourly data

    stock_data = yf.download(stock, period=period, interval=interval)
    print(f"Fetched data for {stock}, interval: {interval}, shape: {stock_data.shape}")
    return stock_data
