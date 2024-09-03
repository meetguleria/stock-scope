from django.contrib import admin
from django.urls import path
from predictions import views

urlpatterns = [
  path('predict_stock/', views.predict_stock, name='predict_stock'),
  path('admin/', admin.site.urls),
]
