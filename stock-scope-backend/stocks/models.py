from django.db import models

# Create your models here.
class Stock(models.Model):
  symbol = models.CharField(max_length=10, unique=True)
  name = models.CharField(max_length=100)
  market_cap = models.BigIntegerField(default=0)
  industry = models.CharField(max_length=100, default='Not Specified')

  def __str__(self):
    return f"{self.name} ({self.symbol})"

class HistoricalData(models.Model):
  stock = models.ForeignKey(Stock, related_name='historical_data', on_delete=models.CASCADE)
  date = models.DateField()
  open_price = models.FloatField()
  high_price = models.FloatField()
  low_price = models.FloatField()
  close_price = models.FloatField()
  adjusted_close = models.FloatField()
  volume = models.BigIntegerField()

  class Meta:
    unique_together = ('stock', 'date')
    ordering = ['-date']

  def __str__(self):
    return f"{self.stock.symbol} on {self.date}"

class Prediction(models.Model):
  stock = models.ForeignKey(Stock, related_name='predictions', on_delete=models.CASCADE)
  date = models.DateField()
  predicted_close = models.FloatField()
  confidence = models.FloatField()

  class Meta:
    unique_together = ('stock', 'date')
    ordering = ['-date']

  def __str__(self):
    return f"Prediction for {self.stock.symbol} on {self.date}"