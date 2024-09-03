import pandas as pd
import numpy as np
import ta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_ta_features(df):
    logging.info("Applying technical analysis features...")
    
    # Ensure there's enough data for a 20-period indicator
    if len(df) >= 20:
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    else:
        logging.warning("Not enough data for 20-period indicators. Filling with NaNs.")
        df['SMA_20'] = np.nan
        df['EMA_20'] = np.nan

    # Ensure there's enough data for a 14-period RSI
    if len(df) >= 14:
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    else:
        logging.warning("Not enough data for 14-period RSI. Filling with NaNs.")
        df['RSI'] = np.nan

    df['MACD'] = ta.trend.macd_diff(df['Close'])

    bb_indicator = ta.volatility.BollingerBands(df['Close'])
    df['BB_upper'] = bb_indicator.bollinger_hband()
    df['BB_middle'] = bb_indicator.bollinger_mavg()
    df['BB_lower'] = bb_indicator.bollinger_lband()

    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)

    logging.info(f"Technical analysis features added: {df.columns[-9:].tolist()}")
    
    # Handle NaNs that result from indicator calculations
    df.bfill(inplace=True)
    
    logging.info(f"Data after filling NaNs: {df.shape}")
    
    return df


def full_feature_engineering_with_outliers_daily(daily_df, lower_bound=1.5, upper_bound=1.5):
    logging.info(f"Starting feature engineering for daily data with shape: {daily_df.shape}")
    
    # Retain the original OHLCV columns
    original_ohlcv_columns = daily_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    logging.info(f"Original OHLCV columns retained with shape: {original_ohlcv_columns.shape}")

    # Feature engineering process
    if 'OutlierFlag' not in daily_df.columns:
        daily_df = add_outlier_flag_daily(daily_df, lower_bound=lower_bound, upper_bound=upper_bound)
        logging.info(f"OutlierFlag added. Data shape: {daily_df.shape}")
    
    daily_df['RollingMean_10'] = daily_df['Close'].rolling(window=10).mean()
    daily_df['RollingStd_10'] = daily_df['Close'].rolling(window=10).std()
    daily_df.bfill(inplace=True)
    daily_df['LogReturn'] = np.log(daily_df['Close'] / daily_df['Close'].shift(1))
    daily_df['DaysSinceLastOutlier'] = (daily_df['OutlierFlag'] * np.arange(len(daily_df))).cummax() - np.arange(len(daily_df))
    daily_df['OutlierFreq_30'] = daily_df['OutlierFlag'].rolling(window=30).sum()

    logging.info(f"Feature engineering completed. Data shape: {daily_df.shape}")

    # Apply technical analysis features
    daily_df = apply_ta_features(daily_df)
    logging.info(f"Technical analysis features applied. Data shape: {daily_df.shape}")
    
    # Append the original OHLCV columns back to the engineered DataFrame
    daily_df = pd.concat([original_ohlcv_columns, daily_df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])], axis=1)
    logging.info(f"Final engineered DataFrame shape (daily): {daily_df.shape}")

    return daily_df

def full_feature_engineering_with_outliers_hourly(hourly_df, lower_bound=1.5, upper_bound=1.5):
    logging.info(f"Starting feature engineering for hourly data with shape: {hourly_df.shape}")
    
    # Retain the original OHLCV columns
    original_ohlcv_columns = hourly_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    logging.info(f"Original OHLCV columns retained with shape: {original_ohlcv_columns.shape}")

    # Feature engineering process
    if 'OutlierFlag' not in hourly_df.columns:
        hourly_df = add_outlier_flag_hourly(hourly_df, lower_bound=lower_bound, upper_bound=upper_bound)
        logging.info(f"OutlierFlag added. Data shape: {hourly_df.shape}")
    
    hourly_df['RollingMean_10'] = hourly_df['Close'].rolling(window=10).mean()
    hourly_df['RollingStd_10'] = hourly_df['Close'].rolling(window=10).std()
    hourly_df.bfill(inplace=True)
    hourly_df['LogReturn'] = np.log(hourly_df['Close'] / hourly_df['Close'].shift(1))
    hourly_df['DaysSinceLastOutlier'] = (hourly_df['OutlierFlag'] * np.arange(len(hourly_df))).cummax() - np.arange(len(hourly_df))
    hourly_df['OutlierFreq_30'] = hourly_df['OutlierFlag'].rolling(window=30).sum()

    logging.info(f"Feature engineering completed. Data shape: {hourly_df.shape}")

    # Apply technical analysis features
    hourly_df = apply_ta_features(hourly_df)
    logging.info(f"Technical analysis features applied. Data shape: {hourly_df.shape}")
    
    # Append the original OHLCV columns back to the engineered DataFrame
    hourly_df = pd.concat([original_ohlcv_columns, hourly_df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])], axis=1)
    logging.info(f"Final engineered DataFrame shape (hourly): {hourly_df.shape}")

    return hourly_df

def add_outlier_flag_daily(daily_df, lower_bound=1.5, upper_bound=1.5):
    logging.info(f"Detecting outliers in daily data with shape: {daily_df.shape}")
    
    Q1 = daily_df.quantile(0.25)
    Q3 = daily_df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound_values = Q1 - lower_bound * IQR
    upper_bound_values = Q3 + upper_bound * IQR
    daily_outlier_flagged = daily_df.copy()
    daily_outlier_flagged['OutlierFlag'] = ((daily_df < lower_bound_values) | (daily_df > upper_bound_values)).any(axis=1).astype(int)
    
    logging.info(f"Outlier detection completed. Data shape: {daily_outlier_flagged.shape}")
    
    return daily_outlier_flagged

def add_outlier_flag_hourly(hourly_df, lower_bound=1.5, upper_bound=1.5):
    logging.info(f"Detecting outliers in hourly data with shape: {hourly_df.shape}")
    
    Q1 = hourly_df.quantile(0.25)
    Q3 = hourly_df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound_values = Q1 - lower_bound * IQR
    upper_bound_values = Q3 + upper_bound * IQR
    hourly_outlier_flagged = hourly_df.copy()
    hourly_outlier_flagged['OutlierFlag'] = ((hourly_df < lower_bound_values) | (hourly_df > upper_bound_values)).any(axis=1).astype(int)
    
    logging.info(f"Outlier detection completed. Data shape: {hourly_outlier_flagged.shape}")
    
    return hourly_outlier_flagged

def apply_full_feature_engineering(data, data_type=None):
    logging.info(f"Applying full feature engineering. Data type: {data_type}, Initial shape: {data.shape}")

    if data_type == 'daily':
        data = full_feature_engineering_with_outliers_daily(data)
    elif data_type == 'hourly':
        data = full_feature_engineering_with_outliers_hourly(data)

    logging.info(f"Feature engineering complete. Data shape after processing: {data.shape}")

    # Ensure all required features are present, including original OHLCV columns
    required_features = [
        'Open', 'High', 'Low', 'Close', 'Volume',  # Original OHLCV features
        'RollingMean_10', 'RollingStd_10', 'LogReturn', 'DaysSinceLastOutlier',
        'OutlierFreq_30', 'SMA_20', 'EMA_20', 'RSI', 'MACD', 'BB_upper',
        'BB_middle', 'BB_lower', 'ATR', 'MFI', 'OutlierFlag'  # Engineered features
    ]

    # Add missing features with default values
    for feature in required_features:
        if feature not in data.columns:
            data[feature] = 0
            logging.warning(f"Feature {feature} missing. Added with default value 0.")

    # Ensure the order of features is consistent
    data = data[required_features]
    logging.info(f"Final DataFrame shape after ensuring all required features: {data.shape}")

    return data
