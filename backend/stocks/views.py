from django.http import JsonResponse
from .model_loader import load_gru_model, load_lstm_model, load_random_forest_model, load_meta_model, load_scaler, load_xgb_model
from .utils import load_latest_stock_data, fetch_and_save_stock_data
from .feature_engineering import apply_feature_engineering_to_stock_data
from datetime import datetime
import numpy as np

def predict_stock(request):
    stock_symbol = request.GET.get('symbol', 'AAPL')  # Default to AAPL

    # Load the latest stock data from the saved CSV
    df_latest = load_latest_stock_data(stock_symbol)
    if df_latest is None:
        return JsonResponse({'error': 'Failed to retrieve stock data'}, status=400)

    # Apply feature engineering to the loaded data
    df_latest_with_features = apply_feature_engineering_to_stock_data(stock_symbol, df_latest)

    # Prepare the input data (use the latest row of features, excluding 'Date' and 'Close')
    latest_row = df_latest_with_features.iloc[-1]
    features_to_use = latest_row.drop(['Date', 'Close'], errors='ignore')

    # Load the scalers
    scaler_X = load_scaler(stock_symbol, 'X')
    scaler_y = load_scaler(stock_symbol, 'y')

    # Scale the input data
    input_data_scaled = scaler_X.transform([features_to_use])

    # Load the models
    gru_model = load_gru_model(stock_symbol)
    lstm_model = load_lstm_model(stock_symbol)
    rf_model = load_random_forest_model(stock_symbol)
    xgb_model = load_xgb_model(stock_symbol)
    meta_model = load_meta_model(stock_symbol)

    # Make predictions using individual models
    gru_prediction = gru_model.predict(input_data_scaled)[0][0]
    lstm_prediction = lstm_model.predict(input_data_scaled)[0][0]
    rf_prediction = rf_model.predict(input_data_scaled)[0]
    xgb_prediction = xgb_model.predict(input_data_scaled)[0]

    # Combine predictions for the meta-model
    predictions = np.array([[gru_prediction, lstm_prediction, rf_prediction, xgb_prediction]])
    final_prediction = meta_model.predict(predictions)[0]

    # Inverse-transform the final prediction to get the original scale
    final_prediction_rescaled = scaler_y.inverse_transform([[final_prediction]])[0][0]

    # Return the predictions in a JSON response
    return JsonResponse({
        'gru_prediction': gru_prediction,
        'lstm_prediction': lstm_prediction,
        'rf_prediction': rf_prediction,
        'xgb_prediction': xgb_prediction,
        'final_prediction': final_prediction_rescaled
    })


def fetch_stock_view(request):
    """Manual endpoint to fetch and save stock data for a given symbol."""
    stock_symbol = request.GET.get('symbol', 'AAPL')

    # Set the date range for fetching the stock data (1-year window for example)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now().replace(year=datetime.now().year - 1)).strftime('%Y-%m-%d')

    # Fetch and save the stock data
    file_path = fetch_and_save_stock_data(stock_symbol, start_date, end_date)

    if file_path:
        return JsonResponse({"message": f"Stock data for {stock_symbol} saved successfully", "file_path": file_path})
    else:
        return JsonResponse({"error": f"Failed to fetch stock data for {stock_symbol}"}, status=400)
