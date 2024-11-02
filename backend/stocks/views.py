import logging
from django.http import JsonResponse
from .ml_pipeline.pipeline import run_forecast_pipeline
from .ml_pipeline.data_management.data_loader import fetch_and_save_stock_data
from datetime import datetime

# Configure the logger for this module
logger = logging.getLogger(__name__)

def predict_stock(request):
    stock_symbol = request.GET.get('symbol', 'AAPL').upper()
    logger.info(f"Received request to predict stock: {stock_symbol}")
    try:
        # Run the forecast pipeline
        predictions = run_forecast_pipeline(stock_symbol)
        logger.info(f"Predictions generated successfully for {stock_symbol}")
        return JsonResponse({
            'status': 'success',
            'predictions': predictions
        })
    except Exception as e:
        # Log the exception with traceback
        logger.exception(f"Error occurred while predicting stock {stock_symbol}: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

def fetch_stock_view(request):
    stock_symbol = request.GET.get('symbol', 'AAPL').upper()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now().replace(year=datetime.now().year - 1)).strftime('%Y-%m-%d')
    logger.info(f"Received request to fetch stock data for: {stock_symbol}, from {start_date} to {end_date}")
    try:
        file_path = fetch_and_save_stock_data(stock_symbol, start_date, end_date)
        logger.info(f"Data for {stock_symbol} saved successfully at {file_path}")
        return JsonResponse({
            "status": "success",
            "message": f"Data for {stock_symbol} saved successfully",
            "file_path": file_path
        })
    except Exception as e:
        logger.exception(f"Error occurred while fetching stock data for {stock_symbol}: {e}")
        return JsonResponse({
            "status": "error",
            "message": str(e)
        }, status=500)
