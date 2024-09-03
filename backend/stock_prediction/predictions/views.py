import numpy as np
import logging
from django.http import JsonResponse
from predictions.utils.models_utils import load_models
from predictions.utils.data_utils import fetch_ohlc_data
from predictions.utils.scaling_utils import scale_dataframe
from predictions.utils.feature_engineering_utils import apply_full_feature_engineering
from predictions.utils.downsampling_utils import downsample_hourly_to_daily, expand_resampled_hourly_preds
from predictions.utils.ensemble_utils import ensemble_predictions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname=s) - %(message=s)')

def predict_stock(request):
    stock_symbol = request.GET.get('stock', 'AAPL')  # Default to 'AAPL' if not provided

    try:
        # Load models
        models = load_models()
        logging.info(f"Models loaded successfully.")

        # Fetch the latest data
        daily_data = fetch_ohlc_data(stock_symbol, period='6mo', interval='1d')
        hourly_data = fetch_ohlc_data(stock_symbol, period='1mo', interval='1h')
        logging.info(f"Fetched data for {stock_symbol}: Daily shape = {daily_data.shape}, Hourly shape = {hourly_data.shape}")

        # Check if data is available
        if daily_data.empty or hourly_data.empty:
            logging.error(f"No data found for {stock_symbol}.")
            return JsonResponse({'error': 'No data found for the selected stock.'}, status=404)

        # Apply feature engineering
        daily_data = apply_full_feature_engineering(daily_data, data_type='daily')
        hourly_data = apply_full_feature_engineering(hourly_data, data_type='hourly')
        logging.info(f"Feature engineering completed: Daily shape = {daily_data.shape}, Hourly shape = {hourly_data.shape}")

        # Scale data for each model
        daily_orig_data_scaled = scale_dataframe(daily_data[['Open', 'High', 'Low', 'Close', 'Volume']], f'{stock_symbol}_daily')
        daily_outlier_data_scaled = scale_dataframe(daily_data, f'{stock_symbol}_daily_outlier')
        daily_ta_data_scaled = scale_dataframe(daily_data, f'{stock_symbol}_daily_ta')

        hourly_orig_data_scaled = scale_dataframe(hourly_data[['Open', 'High', 'Low', 'Close', 'Volume']], f'{stock_symbol}_hourly')
        hourly_outlier_data_scaled = scale_dataframe(hourly_data, f'{stock_symbol}_hourly_outlier')
        hourly_ta_data_scaled = scale_dataframe(hourly_data, f'{stock_symbol}_hourly_ta')

        logging.info(f"Data scaling completed.")

        # Generate ensemble predictions for daily models
        ensemble_preds_daily = ensemble_predictions(
            [models['daily_orig'], models['daily_outlier'], models['daily_ta']],
            [daily_orig_data_scaled[-30:], daily_outlier_data_scaled[-30:], daily_ta_data_scaled[-30:]]
        )

        # Generate ensemble predictions for hourly models
        ensemble_preds_hourly = ensemble_predictions(
            [models['hourly_orig'], models['hourly_outlier'], models['hourly_ta']],
            [hourly_orig_data_scaled[-24:], hourly_outlier_data_scaled[-24:], hourly_ta_data_scaled[-24:]]
        )

        logging.info(f"Ensemble predictions generated: Daily ensemble shape = {ensemble_preds_daily.shape}, Hourly ensemble shape = {ensemble_preds_hourly.shape}")

        # Downsample and expand hourly predictions to match daily predictions
        downsampled_hourly_preds = downsample_hourly_to_daily(ensemble_preds_hourly, len(ensemble_preds_daily))
        expanded_hourly_preds = expand_resampled_hourly_preds(downsampled_hourly_preds, ensemble_preds_daily.shape)
        logging.info(f"Downsampling and expansion completed: Downsampled hourly shape = {downsampled_hourly_preds.shape}, Expanded hourly shape = {expanded_hourly_preds.shape}")

        # Concatenate daily and hourly predictions to create the final input
        final_input = np.concatenate((ensemble_preds_daily[:, np.newaxis], expanded_hourly_preds[:, np.newaxis]), axis=2)
        logging.info(f"Final input shape before expanding dimensions: {final_input.shape}")

        # Ensure the final input has the correct shape (1, 30, 2)
        final_input = np.expand_dims(final_input, axis=0)  # Add batch dimension
        logging.info(f"Final input shape after expanding dimensions: {final_input.shape}")

        # Make the final prediction using the final model
        final_prediction = models['final'].predict(final_input)
        logging.info(f"Final prediction completed: Final prediction shape = {final_prediction.shape}")
        logging.info(f"Final prediction values: {final_prediction}")

        # Return the prediction in JSON format
        return JsonResponse({
            'stock': stock_symbol,
            'predicted_close': float(final_prediction[0, 0])
        })

    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return JsonResponse({'error': str(e)}, status=500)
