from django.urls import path
from . import views

urlpatterns = [
  path('predict/', views.predict_stock, name='predict_stock'),
]