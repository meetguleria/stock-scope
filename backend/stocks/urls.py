from django.urls import path
from . import views

urlpatterns = [
  path('predict/', views.predict_stock, name='predict_stock'),
  path('fetch-stock/', views.fetch_stock_view, name='fetch_stock')
]

