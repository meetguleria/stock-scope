import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# --- Outlier Detection ---

def detect_outliers_and_store(df):
    """
    Detect outliers in the DataFrame using the IQR method and return DataFrames for outliers and clean data.
    """
    if df.empty:
        raise ValueError("The input DataFrame is empty.")

    # Drop any missing values to avoid issues during quantile calculation
    df_cleaned = df.dropna()

    if df_cleaned.empty:
        raise ValueError("The DataFrame contains only NaN values after dropping missing data.")

    # Calculate IQR for each column
    Q1 = df_cleaned.quantile(0.25)
    Q3 = df_cleaned.quantile(0.75)
    IQR = Q3 - Q1

    # Determine outliers across all columns
    outliers = ((df_cleaned < (Q1 - 1.5 * IQR)) | (df_cleaned > (Q3 + 1.5 * IQR))).any(axis=1)
    
    return df_cleaned[outliers], df_cleaned[~outliers]

# --- Daily Data Processing ---

def apply_robust_scaling_daily(daily_df):
    """
    Apply RobustScaler to daily data to scale features while minimizing the impact of outliers.
    """
    if daily_df.empty:
        raise ValueError("The input daily DataFrame is empty.")
    
    scaler = RobustScaler()
    daily_scaled_df = pd.DataFrame(scaler.fit_transform(daily_df), columns=daily_df.columns, index=daily_df.index)
    return daily_scaled_df

def cap_outliers_daily(daily_df, lower_bound=1.5, upper_bound=1.5):
    """
    Cap outliers in daily data using the IQR method.
    """
    if daily_df.empty:
        raise ValueError("The input daily DataFrame is empty.")
    
    Q1 = daily_df.quantile(0.25)
    Q3 = daily_df.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_cap = Q1 - lower_bound * IQR
    upper_cap = Q3 + upper_bound * IQR
    
    daily_capped_df = daily_df.clip(lower=lower_cap, upper=upper_cap, axis=1)
    
    return daily_capped_df

def add_outlier_flag_daily(daily_df, lower_bound=1.5, upper_bound=1.5):
    """
    Add a flag to daily data indicating whether each row contains outliers.
    """
    if daily_df.empty:
        raise ValueError("The input daily DataFrame is empty.")
    
    Q1 = daily_df.quantile(0.25)
    Q3 = daily_df.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound_values = Q1 - lower_bound * IQR
    upper_bound_values = Q3 + upper_bound * IQR
    
    daily_outlier_flagged = daily_df.copy()
    daily_outlier_flagged['OutlierFlag'] = ((daily_df < lower_bound_values) | (daily_df > upper_bound_values)).any(axis=1).astype(int)
    
    return daily_outlier_flagged

def remove_outliers_daily(daily_df, lower_bound=1.5, upper_bound=1.5):
    """
    Remove rows containing outliers in daily data.
    """
    if daily_df.empty:
        raise ValueError("The input daily DataFrame is empty.")
    
    Q1 = daily_df.quantile(0.25)
    Q3 = daily_df.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound_values = Q1 - lower_bound * IQR
    upper_bound_values = Q3 + upper_bound * IQR
    
    daily_no_outliers = daily_df[(daily_df >= lower_bound_values) & (daily_df <= upper_bound_values)].dropna()
    
    return daily_no_outliers

# --- Hourly Data Processing ---

def apply_robust_scaling_hourly(hourly_df):
    """
    Apply RobustScaler to hourly data to scale features while minimizing the impact of outliers.
    """
    if hourly_df.empty:
        raise ValueError("The input hourly DataFrame is empty.")
    
    scaler = RobustScaler()
    hourly_scaled_df = pd.DataFrame(scaler.fit_transform(hourly_df), columns=hourly_df.columns, index=hourly_df.index)
    return hourly_scaled_df

def cap_outliers_hourly(hourly_df, lower_bound=1.5, upper_bound=1.5):
    """
    Cap outliers in hourly data using the IQR method.
    """
    if hourly_df.empty:
        raise ValueError("The input hourly DataFrame is empty.")
    
    Q1 = hourly_df.quantile(0.25)
    Q3 = hourly_df.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_cap = Q1 - lower_bound * IQR
    upper_cap = Q3 + upper_bound * IQR
    
    hourly_capped_df = hourly_df.clip(lower=lower_cap, upper=upper_cap, axis=1)
    
    return hourly_capped_df

def add_outlier_flag_hourly(hourly_df, lower_bound=1.5, upper_bound=1.5):
    """
    Add a flag to hourly data indicating whether each row contains outliers.
    """
    if hourly_df.empty:
        raise ValueError("The input hourly DataFrame is empty.")
    
    Q1 = hourly_df.quantile(0.25)
    Q3 = hourly_df.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound_values = Q1 - lower_bound * IQR
    upper_bound_values = Q3 + upper_bound * IQR
    
    hourly_outlier_flagged = hourly_df.copy()
    hourly_outlier_flagged['OutlierFlag'] = ((hourly_df < lower_bound_values) | (hourly_df > upper_bound_values)).any(axis=1).astype(int)
    
    return hourly_outlier_flagged

def remove_outliers_hourly(hourly_df, lower_bound=1.5, upper_bound=1.5):
    """
    Remove rows containing outliers in hourly data.
    """
    if hourly_df.empty:
        raise ValueError("The input hourly DataFrame is empty.")
    
    Q1 = hourly_df.quantile(0.25)
    Q3 = hourly_df.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound_values = Q1 - lower_bound * IQR
    upper_bound_values = Q3 + upper_bound * IQR
    
    hourly_no_outliers = hourly_df[(hourly_df >= lower_bound_values) & (hourly_df <= upper_bound_values)].dropna()
    
    return hourly_no_outliers
