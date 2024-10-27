from .data_management.data_loader import load_latest_stock_data
from .features_engineering.feature_engineering import apply_feature_engineering_to_stock_data
from .models_management.models_loader import (
    load_xgb_model, load_gru_model, load_lstm_model,
    load_random_forest_model, load_meta_model
)
from .scaling.scalers_loader import load_scalers
from .models_management.predict import make_predictions

def run_forecast_pipeline(stock_symbol):
    # Load and feature-engineer the latest stock data
    df = load_latest_stock_data(stock_symbol)
    df_with_features = apply_feature_engineering_to_stock_data(stock_symbol, df)

    # Load scalers
    scaler_X, scaler_y = load_scalers(stock_symbol)

    # Scale the input data using scaler_X (exclude non-feature columns if necessary)
    features = df_with_features.drop(['Date', 'Close'], axis=1, errors='ignore')
    scaled_data = scaler_X.transform(features)

    # Load each model
    gru_model = load_gru_model(stock_symbol)
    lstm_model = load_lstm_model(stock_symbol)
    rf_model = load_random_forest_model(stock_symbol)
    xgb_model = load_xgb_model(stock_symbol)
    meta_model = load_meta_model(stock_symbol)

    # Generate predictions using each model and pass to the meta-model
    predictions = make_predictions(scaled_data, gru_model, lstm_model, rf_model, xgb_model, meta_model)

    # Rescale final prediction to original target scale
    final_prediction_rescaled = scaler_y.inverse_transform([[predictions['final_prediction']]])[0][0]

    return {
        'gru_prediction': predictions['gru'],
        'lstm_prediction': predictions['lstm'],
        'rf_prediction': predictions['rf'],
        'xgb_prediction': predictions['xgb'],
        'final_prediction': final_prediction_rescaled
    }
