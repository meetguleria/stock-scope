import numpy as np

def make_predictions(scaled_data, gru_model, lstm_model, rf_model, xgb_model, meta_model):
    # Generate individual model predictions on scaled input data
    gru_pred = gru_model.predict(scaled_data)[0][0]   # GRU model prediction
    lstm_pred = lstm_model.predict(scaled_data)[0][0]  # LSTM model prediction
    rf_pred = rf_model.predict(scaled_data)[0]         # Random Forest model prediction
    xgb_pred = xgb_model.predict(scaled_data)[0]       # XGBoost model prediction

    # Combine base model predictions for the meta-model
    combined_predictions = np.array([[gru_pred, lstm_pred, rf_pred, xgb_pred]])
    final_prediction = meta_model.predict(combined_predictions)[0]  # Meta-model prediction

    return {
        'gru': gru_pred,
        'lstm': lstm_pred,
        'rf': rf_pred,
        'xgb': xgb_pred,
        'final_prediction': final_prediction
    }
