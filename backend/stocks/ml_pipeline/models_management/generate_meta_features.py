import logging
import os
import numpy as np
import pandas as pd
from ..scaling.scalers_loader import load_scalers
from ..models_management.models_loader import (
    load_lstm_model,
    load_gru_model,
    load_random_forest_model,
    load_xgboost_model,
    load_feature_names
)

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models'))
TIMESTEPS = 60  # Ensure this matches your training configuration

def create_sequences(data, timesteps):
    sequences = []
    for i in range(len(data) - timesteps + 1):
        sequences.append(data[i:(i + timesteps)])
    return np.array(sequences)

def generate_meta_features(stock_symbol, X_test):
    logger.info(f"Generating meta-features for {stock_symbol}")
    # Initialize meta_features with an index
    meta_features = pd.DataFrame(index=[0])

    try:
        # Load scalers
        scaler_X, scaler_y = load_scalers(stock_symbol)
        if scaler_X is None or scaler_y is None:
            logger.error(f"Scalers not found for {stock_symbol}")
            return None
        logger.debug("Scalers loaded successfully.")

        # Ensure X_test is a DataFrame
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
            logger.debug("Converted X_test to DataFrame.")

        # Check if X_test has enough data points
        if X_test.shape[0] < TIMESTEPS:
            logger.error(f"Not enough data to create sequences for {stock_symbol}. Required: {TIMESTEPS}, available: {X_test.shape[0]}")
            return None
        logger.debug(f"X_test has sufficient data: {X_test.shape[0]} rows.")

        # Prepare data for sequence models (LSTM and GRU)
        X_test_seq = create_sequences(X_test.values, TIMESTEPS)
        logger.debug(f"Input sequences for LSTM/GRU created with shape: {X_test_seq.shape}")

        # Load and predict with LSTM model
        try:
            lstm_model = load_lstm_model(stock_symbol)
            if lstm_model is None:
                raise ValueError("LSTM model failed to load.")
            lstm_predictions_scaled = lstm_model.predict(X_test_seq)
            logger.debug(f"LSTM predictions_scaled shape: {lstm_predictions_scaled.shape}")
            logger.debug(f"LSTM predictions_scaled content: {lstm_predictions_scaled}")
            if lstm_predictions_scaled.size == 0:
                logger.error("LSTM model returned empty predictions.")
                meta_features.at[0, 'LSTM_Pred'] = np.nan
            else:
                meta_features.at[0, 'LSTM_Pred'] = lstm_predictions_scaled.flatten()[-1]  # Take the last prediction
                logger.info(f"LSTM prediction (scaled): {meta_features.at[0, 'LSTM_Pred']}")
        except Exception as e:
            logger.exception(f"Failed to generate LSTM predictions for {stock_symbol}: {e}")
            meta_features.at[0, 'LSTM_Pred'] = np.nan

        # Load and predict with GRU model
        try:
            gru_model = load_gru_model(stock_symbol)
            if gru_model is None:
                raise ValueError("GRU model failed to load.")
            gru_predictions_scaled = gru_model.predict(X_test_seq)
            logger.debug(f"GRU predictions_scaled shape: {gru_predictions_scaled.shape}")
            logger.debug(f"GRU predictions_scaled content: {gru_predictions_scaled}")
            if gru_predictions_scaled.size == 0:
                logger.error("GRU model returned empty predictions.")
                meta_features.at[0, 'GRU_Pred'] = np.nan
            else:
                meta_features.at[0, 'GRU_Pred'] = gru_predictions_scaled.flatten()[-1]  # Take the last prediction
                logger.info(f"GRU prediction (scaled): {meta_features.at[0, 'GRU_Pred']}")
        except Exception as e:
            logger.exception(f"Failed to generate GRU predictions for {stock_symbol}: {e}")
            meta_features.at[0, 'GRU_Pred'] = np.nan

        # Prepare data for non-sequence models (RF and XGBoost)
        X_test_non_seq = X_test.iloc[TIMESTEPS - 1:].reset_index(drop=True)
        logger.debug(f"Data for RF and XGBoost models prepared with shape: {X_test_non_seq.shape}")

        # Load and predict with Random Forest model
        try:
            rf_model = load_random_forest_model(stock_symbol)
            if rf_model is None:
                raise ValueError("Random Forest model failed to load.")
            rf_feature_names = load_feature_names(stock_symbol, model_type='random_forest')
            if rf_feature_names is None:
                raise ValueError("Random Forest feature names failed to load.")
            if not set(rf_feature_names).issubset(X_test_non_seq.columns):
                missing_features = set(rf_feature_names) - set(X_test_non_seq.columns)
                logger.error(f"Random Forest model expects missing features: {missing_features}")
                meta_features.at[0, 'RF_Pred'] = np.nan
            else:
                X_test_rf = X_test_non_seq[rf_feature_names]
                rf_predictions_scaled = rf_model.predict(X_test_rf)
                logger.debug(f"Random Forest predictions_scaled shape: {rf_predictions_scaled.shape}")
                logger.debug(f"Random Forest predictions_scaled content: {rf_predictions_scaled}")
                if rf_predictions_scaled.size == 0:
                    logger.error("Random Forest model returned empty predictions.")
                    meta_features.at[0, 'RF_Pred'] = np.nan
                else:
                    meta_features.at[0, 'RF_Pred'] = rf_predictions_scaled[-1]  # Take the last prediction
                    logger.info(f"Random Forest prediction (scaled): {meta_features.at[0, 'RF_Pred']}")
        except Exception as e:
            logger.exception(f"Failed to generate Random Forest predictions for {stock_symbol}: {e}")
            meta_features.at[0, 'RF_Pred'] = np.nan

        # Load and predict with XGBoost model
        try:
            xgb_model = load_xgboost_model(stock_symbol)
            if xgb_model is None:
                raise ValueError("XGBoost model failed to load.")
            xgb_feature_names = load_feature_names(stock_symbol, model_type='xgboost')
            if xgb_feature_names is None:
                raise ValueError("XGBoost feature names failed to load.")
            if not set(xgb_feature_names).issubset(X_test_non_seq.columns):
                missing_features = set(xgb_feature_names) - set(X_test_non_seq.columns)
                logger.error(f"XGBoost model expects missing features: {missing_features}")
                meta_features.at[0, 'XGB_Pred'] = np.nan
            else:
                X_test_xgb = X_test_non_seq[xgb_feature_names]
                xgb_predictions_scaled = xgb_model.predict(X_test_xgb)
                logger.debug(f"XGBoost predictions_scaled shape: {xgb_predictions_scaled.shape}")
                logger.debug(f"XGBoost predictions_scaled content: {xgb_predictions_scaled}")
                if xgb_predictions_scaled.size == 0:
                    logger.error("XGBoost model returned empty predictions.")
                    meta_features.at[0, 'XGB_Pred'] = np.nan
                else:
                    meta_features.at[0, 'XGB_Pred'] = xgb_predictions_scaled[-1]  # Take the last prediction
                    logger.info(f"XGBoost prediction (scaled): {meta_features.at[0, 'XGB_Pred']}")
        except Exception as e:
            logger.exception(f"Failed to generate XGBoost predictions for {stock_symbol}: {e}")
            meta_features.at[0, 'XGB_Pred'] = np.nan

        # Check if any predictions failed
        if meta_features.isnull().values.any():
            logger.warning("Some meta-feature predictions contain NaN values.")

        # Verify if meta_features has the required columns
        required_columns = ['LSTM_Pred', 'GRU_Pred', 'RF_Pred', 'XGB_Pred']
        missing_columns = [col for col in required_columns if col not in meta_features.columns]
        if missing_columns:
            logger.error(f"Missing meta-feature columns: {missing_columns}")
            return None

        # Return the meta-features DataFrame
        logger.info(f"Meta-features generation completed for {stock_symbol}")
        return meta_features

    except Exception as e:
        logger.exception(f"Error in generating meta-features for {stock_symbol}: {e}")
        return None
