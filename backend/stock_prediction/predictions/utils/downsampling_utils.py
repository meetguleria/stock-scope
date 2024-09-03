import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def downsample_hourly_to_daily(hourly_preds, daily_length):

    logging.info(f"Downsampling hourly predictions from length {len(hourly_preds)} to match daily length {daily_length}.")

    # Calculate the step size needed to downsample
    step_size = len(hourly_preds) // daily_length

    if step_size == 0:
        logging.error("Step size is zero, cannot downsample correctly.")
        return np.array([])  # Return empty array to avoid errors

    # Use slicing to downsample
    downsampled_preds = hourly_preds[::step_size]

    # Ensure the downsampled data has the exact length as daily data
    downsampled_preds = downsampled_preds[:daily_length]

    logging.info(f"Downsampled predictions shape: {downsampled_preds.shape}")

    return downsampled_preds

def expand_resampled_hourly_preds(resampled_hourly_preds, daily_shape):

    logging.info(f"Expanding resampled hourly predictions to match daily data shape {daily_shape}.")

    # Repeat the resampled hourly predictions to match the time steps in daily data
    ensemble_preds_hourly_expanded = np.repeat(resampled_hourly_preds[:, np.newaxis], daily_shape[1], axis=1)

    logging.info(f"Expanded hourly predictions shape: {ensemble_preds_hourly_expanded.shape}")
    
    return ensemble_preds_hourly_expanded
