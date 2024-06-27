# Stock Scope

## Project Overview
Stock Scope is a comprehensive tool for analyzing historical stock data, integrating technical indicators, financial metrics, and detecting anomalies. This project aims to provide insights into stock market trends and support quantitative analysis.

## Features
- **Data Preprocessing and Cleaning**: Handles missing values, outliers, and feature engineering.
- **Technical Indicators**: Incorporates moving averages, volume averages, EMA, RSI, MACD, and Bollinger Bands.
- **Anomaly Detection**: Identifies high-volume anomalies in stock data.
- **Financial Ratios and Metrics**: Computes key financial ratios such as P/E ratio, Debt-to-Equity ratio, and Return on Equity.
- **Exploratory Data Analysis (EDA)**: Visualizes stock data trends, correlations, and distributions.
- **Machine Learning Models**: Implements predictive models to forecast stock prices and returns.

## Data Sources
Stock data is fetched from Yahoo Finance using the `yfinance` library, which includes:
- **Historical Stock Prices**: Open, High, Low, Close, Volume
- **Dividends and Stock Splits**
- **Financial Statements**: Income Statement, Balance Sheet, Cash Flow
- **Financial Ratios**: P/E Ratio, Debt-to-Equity, ROE, etc.

## Installation

### Clone the Repository
```sh
git clone https://github.com/yourusername/stock-scope.git
cd stock-scope
```

### Create and Activate a Virtual Environment
```sh
pyenv virtualenv 3.12.4 stock-scope-env
pyenv local stock-scope-env
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Usage

### Fetch Stock Data
To fetch and save stock data, run the `fetch_data.py` script:
```sh
python downloads/fetch_data.py
```

### Jupyter Notebooks
Navigate to the `notebooks` directory and start Jupyter Notebook to explore and analyze the data:
```sh
jupyter notebook
```

## Data Collection
Historical stock data is fetched using the `yfinance` library. The following types of data are collected:

- Daily stock prices (Open, High, Low, Close, Volume)
- Dividends and stock splits
- Key financial metrics such as P/E ratio, earnings, and more.

### Script for Data Collection
A script named `fetch_data.py` is provided to automate data collection. It saves the data in CSV format in the `downloads` directory.

## Data Preprocessing and Cleaning
Data preprocessing includes handling missing values, cleaning anomalies, and normalizing data. We ensure the data is ready for feature engineering and model training.

## Feature Engineering
Various technical indicators and financial metrics are added to enhance the predictive power of our models. This includes:

- Moving Averages (20-day, 50-day, 200-day)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- Volume-related metrics

## Exploratory Data Analysis (EDA)
EDA is performed to understand the data distribution, trends, and correlations. Key steps include:

- Visualizing historical stock prices
- Analyzing volume anomalies
- Plotting technical indicators
- Correlation matrix to identify relationships between features

## Machine Learning Models
Various machine learning models are implemented to predict stock prices and identify market trends:

- Linear Regression
- Decision Trees
- Random Forests
- Support Vector Machines
- Neural Networks
- LSTM Neural Networks

## Anomaly Detection
We detect high-volume anomalies to identify unusual trading activity, which might indicate significant events affecting stock prices. The detection process involves:

- Calculating mean and standard deviation of volume
- Identifying days with volumes significantly higher than the mean

## Results
The project includes various visualizations and models for stock data analysis, including:
- **Correlation Heatmaps**
- **High-Volume Anomaly Charts**
- **Financial Ratio Analysis**
- **Predictive Modeling Outputs**

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests.

## License
This project is licensed under the MIT License.

## Acknowledgments
- The `yfinance` library for data fetching
- The `ta` library for technical analysis indicators
- Contributors and the open-source community
```