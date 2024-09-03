import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i+sequence_length, :]
        target = data[i+sequence_length, 0]
        sequences.append(sequence)
        targets.append(target)
    logging.info(f"Created sequences with shape: {np.array(sequences).shape} and targets with shape: {np.array(targets).shape}")
    return np.array(sequences), np.array(targets)

def build_lstm_model(input_shape):
    logging.info(f"Building LSTM model with input shape: {input_shape}")
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    logging.info("LSTM model built and compiled successfully.")
    return model

def train_and_save_model(X_train, y_train, X_test, y_test, model_name, sequence_length, save_dir):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    logging.info(f"Training model: {model_name}")
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    loss = model.evaluate(X_test, y_test)
    logging.info(f'Model Loss ({model_name}): {loss}')
    model_path = os.path.join(save_dir, f'{model_name}.keras')
    model.save(model_path)
    logging.info(f'Model saved at: {model_path}')
    return model

def aggregate_and_train_models(stocks, scaled_daily_data_dict, scaled_daily_outlier_features_dict, 
                               scaled_daily_ta_features_dict, scaled_hourly_data_dict, 
                               scaled_hourly_outlier_features_dict, scaled_hourly_ta_features_dict, save_dir):
    # Sequence lengths
    sequence_length_daily = 30  # 30 days
    sequence_length_hourly = 24  # 24 hours

    # Aggregated data across all stocks for each type of model
    aggregated_daily_orig = []
    aggregated_daily_outlier = []
    aggregated_daily_ta = []
    aggregated_hourly_orig = []
    aggregated_hourly_outlier = []
    aggregated_hourly_ta = []

    # Aggregate data across all stocks
    for stock in stocks:
        aggregated_daily_orig.extend(scaled_daily_data_dict[stock].values)
        aggregated_daily_outlier.extend(scaled_daily_outlier_features_dict[stock].values)
        aggregated_daily_ta.extend(scaled_daily_ta_features_dict[stock].values)
        
        aggregated_hourly_orig.extend(scaled_hourly_data_dict[stock].values)
        aggregated_hourly_outlier.extend(scaled_hourly_outlier_features_dict[stock].values)
        aggregated_hourly_ta.extend(scaled_hourly_ta_features_dict[stock].values)

    # Convert lists to numpy arrays
    aggregated_daily_orig = np.array(aggregated_daily_orig)
    aggregated_daily_outlier = np.array(aggregated_daily_outlier)
    aggregated_daily_ta = np.array(aggregated_daily_ta)
    aggregated_hourly_orig = np.array(aggregated_hourly_orig)
    aggregated_hourly_outlier = np.array(aggregated_hourly_outlier)
    aggregated_hourly_ta = np.array(aggregated_hourly_ta)

    # Train and save models
    X_daily_orig, y_daily_orig = create_sequences(aggregated_daily_orig, sequence_length_daily)
    X_train_daily_orig, X_test_daily_orig, y_train_daily_orig, y_test_daily_orig = train_test_split(X_daily_orig, y_daily_orig, test_size=0.2, shuffle=False)
    train_and_save_model(X_train_daily_orig, y_train_daily_orig, X_test_daily_orig, y_test_daily_orig, 
                         'model_daily_orig', sequence_length_daily, save_dir)
    
    X_daily_outlier, y_daily_outlier = create_sequences(aggregated_daily_outlier, sequence_length_daily)
    X_train_daily_outlier, X_test_daily_outlier, y_train_daily_outlier, y_test_daily_outlier = train_test_split(X_daily_outlier, y_daily_outlier, test_size=0.2, shuffle=False)
    train_and_save_model(X_train_daily_outlier, y_train_daily_outlier, X_test_daily_outlier, y_test_daily_outlier, 
                         'model_daily_outlier', sequence_length_daily, save_dir)
    
    X_daily_ta, y_daily_ta = create_sequences(aggregated_daily_ta, sequence_length_daily)
    X_train_daily_ta, X_test_daily_ta, y_train_daily_ta, y_test_daily_ta = train_test_split(X_daily_ta, y_daily_ta, test_size=0.2, shuffle=False)
    train_and_save_model(X_train_daily_ta, y_train_daily_ta, X_test_daily_ta, y_test_daily_ta, 
                         'model_daily_ta', sequence_length_daily, save_dir)
    
    X_hourly_orig, y_hourly_orig = create_sequences(aggregated_hourly_orig, sequence_length_hourly)
    X_train_hourly_orig, X_test_hourly_orig, y_train_hourly_orig, y_test_hourly_orig = train_test_split(X_hourly_orig, y_hourly_orig, test_size=0.2, shuffle=False)
    train_and_save_model(X_train_hourly_orig, y_train_hourly_orig, X_test_hourly_orig, y_test_hourly_orig, 
                         'model_hourly_orig', sequence_length_hourly, save_dir)
    
    X_hourly_outlier, y_hourly_outlier = create_sequences(aggregated_hourly_outlier, sequence_length_hourly)
    X_train_hourly_outlier, X_test_hourly_outlier, y_train_hourly_outlier, y_test_hourly_outlier = train_test_split(X_hourly_outlier, y_hourly_outlier, test_size=0.2, shuffle=False)
    train_and_save_model(X_train_hourly_outlier, y_train_hourly_outlier, X_test_hourly_outlier, y_test_hourly_outlier, 
                         'model_hourly_outlier', sequence_length_hourly, save_dir)
    
    X_hourly_ta, y_hourly_ta = create_sequences(aggregated_hourly_ta, sequence_length_hourly)
    X_train_hourly_ta, X_test_hourly_ta, y_train_hourly_ta, y_test_hourly_ta = train_test_split(X_hourly_ta, y_hourly_ta, test_size=0.2, shuffle=False)
    train_and_save_model(X_train_hourly_ta, y_train_hourly_ta, X_test_hourly_ta, y_test_hourly_ta, 
                         'model_hourly_ta', sequence_length_hourly, save_dir)

    logging.info("Training and saving models completed.")
