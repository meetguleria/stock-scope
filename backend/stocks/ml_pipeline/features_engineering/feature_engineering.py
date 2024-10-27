import pandas as pd
import numpy as np
import logging

def add_close_price_features(df):
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['SMA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False, min_periods=1).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False, min_periods=1).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False, min_periods=1).mean()

    # Momentum Indicators
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['ROC_5'] = df['Close'].pct_change(periods=5)
    df['ROC_10'] = df['Close'].pct_change(periods=10)

    # Volatility Indicators
    df['Volatility_5'] = df['Close'].rolling(window=5, min_periods=1).std()
    df['Volatility_10'] = df['Close'].rolling(window=10, min_periods=1).std()

    # RSI Calculation
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.rolling(window=14, min_periods=1).mean()
    average_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = average_gain / (average_loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
    df['MACD_diff'] = df['MACD'] - df['MACD_signal']

    # Bollinger Bands
    df['Middle_Band'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20, min_periods=1).std()
    df['Upper_Band'] = df['Middle_Band'] + (df['Std_Dev'] * 2)
    df['Lower_Band'] = df['Middle_Band'] - (df['Std_Dev'] * 2)
    df['Bollinger_Width'] = df['Upper_Band'] - df['Lower_Band']

    # Percent B (%B)
    df['Percent_B'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'] + 1e-10)

    # Williams %R
    df['Highest_Close_14'] = df['Close'].rolling(window=14, min_periods=1).max()
    df['Lowest_Close_14'] = df['Close'].rolling(window=14, min_periods=1).min()
    df['Williams_%R'] = ((df['Highest_Close_14'] - df['Close']) / (df['Highest_Close_14'] - df['Lowest_Close_14'] + 1e-10)) * -100

    # Exponential Moving Average Differences
    df['EMA_5_10_Diff'] = df['EMA_5'] - df['EMA_10']
    df['EMA_5_20_Diff'] = df['EMA_5'] - df['EMA_20']

    # Lag Features
    df['Lag_Close_1'] = df['Close'].shift(1)
    df['Lag_Close_2'] = df['Close'].shift(2)
    df['Lag_Close_3'] = df['Close'].shift(3)

    # Rolling statistics
    df['Rolling_Skew_Close_5'] = df['Close'].rolling(window=5, min_periods=1).skew()
    df['Rolling_Kurt_Close_5'] = df['Close'].rolling(window=5, min_periods=1).kurt()

    # Fill any missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df

def apply_feature_engineering_to_stock_data(stock_symbol, df):
    logging.info(f"Applying feature engineering for {stock_symbol}")
    df_with_features = add_close_price_features(df)
    logging.info(f"Feature engineering complete for {stock_symbol}")
    return df_with_features

