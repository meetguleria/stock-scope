## Stock Scope

**Stock Scope** is a comprehensive stock forecasting project that integrates a rigorous research pipeline with a production-ready Django backend. The project uses advanced time series analysis, feature engineering, and machine learning to predict stock prices. It leverages deep learning models, ensemble methods, and a meta-model stacking approach to provide robust predictions, all while making these forecasts accessible through RESTful API endpoints.

---

## Overview

Stock Scope is structured into two main components:

1. **Research Pipeline**
   This component handles the end-to-end process of stock prediction:
   - **Data Acquisition & Preprocessing:** Historical stock data is fetched from financial APIs, cleaned, and stored.
   - **Feature Engineering:** A wide range of technical indicators are computed (e.g., moving averages, momentum, volatility, RSI, MACD, Bollinger Bands, etc.) to enhance the raw data.
   - **Model Training & Evaluation:** Several models are trained:
     - **Deep Learning Models:** LSTM and GRU networks to capture sequential dependencies.
     - **Ensemble Models:** Random Forest and XGBoost to model non-linear patterns.
     - **Meta-Model Stacking:** A Ridge regression meta-model combines predictions from the base models to improve overall accuracy.
   - **Forecasting:** An iterative forecasting process updates the input data with new predictions, recalculates technical indicators, and produces future stock price forecasts.

2. **Django Backend**
   This component serves the research insights via a RESTful API:
   - **Prediction Endpoint:** Receives a stock symbol and returns predictions from individual base models (after inverse scaling) along with a final meta-model forecast.
   - **Data Fetching Endpoint:** Fetches the latest stock data and updates stored CSV files.
   - **Model & Scaler Management:** Loads pre-trained models (from Keras, Scikit-learn, and XGBoost) and scalers (MinMaxScalers) to ensure that data passed during inference is consistent with the training process.
   - **Pipeline Integration:** Integrates all components (data loading, feature engineering, scaling, meta-feature generation, and final prediction) to return a structured prediction output.

---

## Main Libraries and Tools

- **Data Handling & Analysis:**
  - **Pandas & NumPy:** For data manipulation, cleaning, and numerical operations.
  - **yfinance:** To download historical stock data.

- **Visualization:**
  - **Matplotlib & Seaborn:** For exploratory data analysis and visualizing trends (primarily used during research).

- **Machine Learning & Deep Learning:**
  - **TensorFlow/Keras:** For building and training deep learning models (LSTM, GRU).
  - **Scikit-learn:** For scaling data, evaluating models (RMSE, MAE, R², MAPE), and implementing the meta-model (RidgeCV).
  - **XGBoost:** For training gradient boosting models.
  - **Joblib & Pickle:** For model and scaler serialization.

- **Backend Framework:**
  - **Django:** Provides the web framework to create RESTful API endpoints that serve predictions and manage data.

---

## Research Pipeline

### Data Acquisition & Preprocessing

- **Data Loader:**
  Historical stock data is downloaded using the `yfinance` library and stored as CSV files. The data is cleaned (e.g., converting columns to numeric types, handling missing values) and sorted by date.

- **Feature Engineering:**  
  The pipeline computes an extensive set of technical indicators including:
  - **Moving Averages:** Simple (SMA) and Exponential (EMA) over different time windows.
  - **Momentum & Rate of Change (ROC):** Capturing short-term trends.
  - **Volatility Measures:** Rolling standard deviations.
  - **Oscillators:** RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence with signal and difference), Bollinger Bands with %B, and Williams %R.
  - **Lag Features & Rolling Statistics:** Including lagged close prices, skewness, and kurtosis.

- **Data Scaling:**
  The data is split into training and test sets using a time-aware split. MinMaxScalers (persisted using joblib) are used to normalize features and target variables, ensuring consistency during both training and inference.

### Model Training & Evaluation

- **Deep Learning Models:**
  LSTM and GRU networks are built using Keras. These models capture sequential patterns in time series data. Techniques such as early stopping and checkpointing are used to improve training.

- **Ensemble Models:**
  Random Forest and XGBoost models are trained on the engineered features to capture non-sequential relationships.

- **Meta-Model Stacking:**
  A Ridge regression model (using RidgeCV) is trained to combine predictions from the LSTM, GRU, Random Forest, and XGBoost models. This stacked model leverages the strengths of each base model to yield more accurate forecasts.

- **Evaluation:**
  Models are evaluated using standard metrics such as RMSE, MAE, R², and MAPE. In addition, performance is analyzed across different time segments (monthly, quarterly, seasonally) to assess robustness.

### Forecasting Process

- **Iterative Forecasting:**
  Future stock prices are forecasted iteratively:
  - The latest available data is updated with the previous prediction.
  - Technical indicators are recalculated after each update.
  - Base models produce new predictions which are then combined by the meta-model.

- **Meta-Feature Generation:**
  Predictions from individual base models are combined into a meta-feature vector that serves as input to the meta-model for the final forecast.

---

## Django Backend

### API Endpoints

- **Predict Stock Endpoint:**  
  - **URL:** `/stocks/predict-stock/`  
  - **Function:** Receives a stock symbol (defaulting to AAPL) and triggers the forecast pipeline.  
  - **Response:** Returns a JSON containing:
    - Predictions from each base model (GRU, LSTM, RF, XGBoost) after inverse scaling.
    - The final aggregated prediction from the meta-model.

- **Fetch Stock Data Endpoint:**  
  - **URL:** `/stocks/fetch-stock/`  
  - **Function:** Fetches and updates the latest stock data using yfinance, saving the data to CSV files.

### Model and Scaler Management

- **Model Loading:**  
  The backend loads models from saved files:
  - **Keras Models (LSTM & GRU):** Loaded using TensorFlow’s `load_model()`.
  - **Random Forest & XGBoost Models:** Loaded using joblib and XGBoost’s native methods, respectively.
  - **Meta-Model:** Loaded via joblib.

- **Scaler Loading:**  
  Feature and target scalers (MinMaxScalers) are loaded to ensure that data is transformed consistently during prediction and then inverse transformed to return results in the original scale.

- **Feature Name Alignment:**  
  During training, the feature names used are stored (using pickle) and are reloaded during inference to ensure that input data aligns correctly with the trained models.

### Pipeline Integration

- **Forecast Pipeline:**  
  The `pipeline.py` module ties together:
  - Data loading and preprocessing.
  - Feature engineering and scaling.
  - Meta-feature generation from base model predictions.
  - Final prediction generation using the meta-model.

- **Utility Functions:**  
  Additional modules handle data fetching, scaler management, and model loading to maintain consistency across the entire forecasting process.

---

## How It Works

1. **Data Acquisition & Preprocessing:**  
   Historical stock data is fetched via yfinance and preprocessed with extensive feature engineering. This process calculates technical indicators that enrich the dataset.

2. **Model Training:**  
   Multiple models are trained on this engineered dataset:
   - LSTM and GRU capture sequential trends.
   - Random Forest and XGBoost capture non-sequential patterns.
   - A Ridge regression meta-model stacks the outputs of these models to improve prediction accuracy.

3. **Forecasting:**  
   Using an iterative process, new predictions are generated by:
   - Updating the latest data with the previous day’s forecast.
   - Recomputing technical indicators.
   - Generating individual base model predictions.
   - Combining these predictions via the meta-model to produce the final forecast.

4. **Backend Integration:**  
   The Django backend exposes these predictions through RESTful API endpoints. It manages the loading of models and scalers, ensures feature alignment, and serves predictions in real time.

---

## Future Enhancements
- **Enhanced Backend Functionality:**  
  Expand the API with better error handling, caching, and potential user authentication.
- **Front-End Development:**  
  Develop a user-friendly front-end (e.g., with React) to visualize forecasts and interact with the prediction system.
- **Additional Model Tuning:**  
  Explore additional machine learning techniques and hyperparameter tuning to further improve forecast accuracy.
---