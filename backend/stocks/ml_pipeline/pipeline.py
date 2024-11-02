import logging
import pandas as pd
from .data_management.data_loader import load_latest_stock_data
from .features_engineering.feature_engineering import apply_feature_engineering_to_stock_data
from .scaling.scalers_loader import load_scalers
from .models_management.models_loader import load_meta_model, load_meta_feature_names
from .models_management.generate_meta_features import generate_meta_features

logger = logging.getLogger(__name__)

def run_forecast_pipeline(stock_symbol):
    logger.info(f"Starting forecast pipeline for {stock_symbol}")

    try:
        # Load the latest stock data
        df = load_latest_stock_data(stock_symbol)
        if df is None or df.empty:
            logger.error(f"No data available for {stock_symbol}")
            return {
                'status': 'error',
                'message': f'No data available for {stock_symbol}'
            }
        logger.info(f"Data loaded for {stock_symbol}, shape: {df.shape}")

        # Apply feature engineering
        df_with_features = apply_feature_engineering_to_stock_data(stock_symbol, df)
        if df_with_features is None or df_with_features.empty:
            logger.error(f"Feature engineering failed for {stock_symbol}")
            return {
                'status': 'error',
                'message': f'Feature engineering failed for {stock_symbol}'
            }
        logger.info(f"Feature engineering applied, shape: {df_with_features.shape}")

        # Load scalers
        scaler_X, scaler_y = load_scalers(stock_symbol)
        if scaler_X is None or scaler_y is None:
            logger.error(f"Scalers not found for {stock_symbol}")
            return {
                'status': 'error',
                'message': f'Scalers not found for {stock_symbol}'
            }
        logger.info(f"Scalers loaded for {stock_symbol}")

        # Prepare features
        features = df_with_features.drop(['Date', 'Close'], axis=1, errors='ignore')
        TIMESTEPS = 60
        features = features.iloc[-(TIMESTEPS + 1):]
        logger.info(f"Features prepared, shape: {features.shape}")

        if hasattr(scaler_X, 'feature_names_in_'):
            expected_feature_names = scaler_X.feature_names_in_
            features = features.reindex(columns=expected_feature_names)
            logger.info("Features reindexed to match expected feature names")
        else:
            logger.warning("Scaler does not have 'feature_names_in_' attribute")

        # Scale features
        scaled_data = scaler_X.transform(features)
        scaled_data = pd.DataFrame(scaled_data, columns=features.columns, index=features.index)
        logger.info(f"Features scaled, shape: {scaled_data.shape}")

        # Generate meta-features using base models
        meta_features_test = generate_meta_features(
            stock_symbol,
            X_test=scaled_data
        )
        if meta_features_test is None or meta_features_test.empty:
            logger.error("Meta-features generation failed")
            return {
                'status': 'error',
                'message': 'Meta-features generation failed'
            }
        logger.info(f"Meta-features generated, shape: {meta_features_test.shape}")

        # Load meta-feature names
        meta_feature_names = load_meta_feature_names(stock_symbol)
        if meta_feature_names is None:
            logger.error("Meta-feature names not found")
            return {
                'status': 'error',
                'message': 'Meta-feature names not found'
            }
        logger.info(f"Meta-feature names loaded: {meta_feature_names}")

        # Align meta-features with training feature names
        logger.debug(f"Training meta-feature names: {meta_feature_names}")
        logger.debug(f"Generated meta-feature columns: {list(meta_features_test.columns)}")

        try:
            meta_features_test = meta_features_test[meta_feature_names]
            logger.debug(f"Meta-features aligned, columns: {list(meta_features_test.columns)}")
        except KeyError as e:
            logger.error(f"Feature alignment failed: {e}")
            return {
                'status': 'error',
                'message': f'Feature alignment failed: {e}'
            }

        # Load the meta-model
        meta_model = load_meta_model(stock_symbol)
        if meta_model is None:
            logger.error("Meta-model not found")
            return {
                'status': 'error',
                'message': 'Meta-model not found'
            }
        logger.info("Meta-model loaded successfully")

        # Make the final prediction
        try:
            final_prediction_scaled = meta_model.predict(meta_features_test)[0]
            logger.debug(f"Scaled Final Prediction: {final_prediction_scaled}")

            # Inverse transform to get the original scale
            final_prediction = scaler_y.inverse_transform([[final_prediction_scaled]])[0][0]
            logger.debug(f"Final Prediction (Inverse Scaled): {final_prediction}")
            logger.info(f"Final prediction: {final_prediction}")
        except Exception as e:
            logger.exception(f"Error making final prediction: {e}")
            return {
                'status': 'error',
                'message': f'Error making final prediction: {e}'
            }

        # Inverse transform individual base model predictions
        for model in ['GRU_Pred', 'LSTM_Pred', 'RF_Pred', 'XGB_Pred']:
            pred_scaled = meta_features_test.at[0, model]
            if pd.notnull(pred_scaled):
                pred_original = scaler_y.inverse_transform([[pred_scaled]])[0][0]
                meta_features_test.at[0, model] = pred_original
                logger.info(f"{model} prediction (inverse scaled): {pred_original}")
            else:
                meta_features_test.at[0, model] = None  # Handle NaN appropriately

        # Prepare the response with native Python floats
        response = {}
        for feature in ['GRU_Pred', 'LSTM_Pred', 'RF_Pred', 'XGB_Pred']:
            pred_value = meta_features_test.at[0, feature]
            response_key = feature.lower() + '_prediction'
            response[response_key] = float(pred_value) if pd.notnull(pred_value) else None

        response['final_prediction'] = float(final_prediction) if pd.notnull(final_prediction) else None
        logger.info("Forecast pipeline completed successfully")

        # Return the response with correct structure
        return {
            'status': 'success',
            'predictions': response
        }

    except Exception as e:
        logger.exception(f"Error in forecast pipeline: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }
