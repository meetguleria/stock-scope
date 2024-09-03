import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname=s) - %(message=s)')

def predict_daily_orig(model, X_test):
    X_test = adjust_input_shape(X_test)
    pred = model.predict(X_test)
    logging.info(f'Prediction shape for daily original model: {pred.shape}')
    return pred

def predict_daily_outlier(model, X_test):
    X_test = adjust_input_shape(X_test)
    pred = model.predict(X_test)
    logging.info(f'Prediction shape for daily outlier model: {pred.shape}')
    return pred

def predict_daily_ta(model, X_test):
    X_test = adjust_input_shape(X_test)
    pred = model.predict(X_test)
    logging.info(f'Prediction shape for daily TA model: {pred.shape}')
    return pred

def predict_hourly_orig(model, X_test):
    X_test = adjust_input_shape(X_test)
    pred = model.predict(X_test)
    logging.info(f'Prediction shape for hourly original model: {pred.shape}')
    return pred

def predict_hourly_outlier(model, X_test):
    X_test = adjust_input_shape(X_test)
    pred = model.predict(X_test)
    logging.info(f'Prediction shape for hourly outlier model: {pred.shape}')
    return pred

def predict_hourly_ta(model, X_test):
    X_test = adjust_input_shape(X_test)
    pred = model.predict(X_test)
    logging.info(f'Prediction shape for hourly TA model: {pred.shape}')
    return pred


def adjust_input_shape(X_test):
    # Ensure X_test is a numpy array
    if isinstance(X_test, tuple):
        X_test = X_test[0]  # Extract the first element if it's a tuple

    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.to_numpy()  # Convert DataFrame to numpy array

    logging.info(f"X_test shape before adjustment: {X_test.shape}")
    if len(X_test.shape) == 2:
        X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
    elif len(X_test.shape) == 3 and X_test.shape[0] != 1:
        X_test = X_test.reshape(1, X_test.shape[1], X_test.shape[2])
    logging.info(f"X_test shape after adjustment: {X_test.shape}")
    return X_test

def ensemble_predictions(models, X_tests):
    ensemble_pred = np.zeros_like(models[0].predict(adjust_input_shape(X_tests[0])))
    for model, X_test in zip(models, X_tests):
        X_test_adjusted = adjust_input_shape(X_test)  # Adjust input shape
        ensemble_pred += model.predict(X_test_adjusted)
    ensemble_pred /= len(models)  # Average the predictions
    return np.squeeze(ensemble_pred)

def evaluate_ensemble(y_true, y_pred, title="Ensemble Model"):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    direction_correct = (np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(y_true[1:] - y_true[:-1])).mean()
    
    logging.info(f'{title} Evaluation:')
    logging.info(f'MSE: {mse}')
    logging.info(f'MAE: {mae}')
    logging.info(f'RMSE: {rmse}')
    logging.info(f'MAPE: {mape}')
    logging.info(f'Directional Accuracy: {direction_correct}')
