from django.shortcuts import render
from .models import Stock

def index(request):
  stocks = Stock.objects.all()
  return render(request, 'stocks/index.html', {'stocks': stocks})