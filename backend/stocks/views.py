from django.http import JsonResponse
from .ml_pipeline.pipeline import run_forecast_pipeline
from .ml_pipeline.data_management.data_loader import fetch_and_save_stock_data  # Adjust import as needed
from datetime import datetime

def predict_stock(request):
    stock_symbol = request.GET.get('symbol', 'AAPL')
    try:
        predictions = run_forecast_pipeline(stock_symbol)
        return JsonResponse({
            'status': 'success',
            'predictions': predictions
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

def fetch_stock_view(request):
    stock_symbol = request.GET.get('symbol', 'AAPL')
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now().replace(year=datetime.now().year - 1)).strftime('%Y-%m-%d')
    try:
        file_path = fetch_and_save_stock_data(stock_symbol, start_date, end_date)
        return JsonResponse({
            "status": "success",
            "message": f"Data for {stock_symbol} saved successfully",
            "file_path": file_path
        })
    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": str(e)
        }, status=500)
