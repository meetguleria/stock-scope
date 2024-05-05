from django.db import models
class Stock(models.Model):
  symbol = models.CharField(max_length=10, unique=True)
  name = models.CharField(max_length=100)
  market_cap = models.BigIntegerField(default=0)
  industry = models.CharField(max_length=100, default='Not Specified')
  sector = models.CharField(max_length=100, default='Not Specified')
  full_time_employees = models.IntegerField(default=0)
  long_business_summary = models.TextField(default='')

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

class Dividends(models.Model):
  stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='dividends')
  date = models.DateField()
  amount = models.FloatField()

  class Meta:
    unique_together = ('stock', 'date')

  def __str__(self):
    return f"Dividend {self.amount} for {self.stock.symbol} on {self.date}"

class Financials(models.Model):
  stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='financials')
  date = models.DateField()
  net_income = models.BigIntegerField(null=True, blank=True)
  total_revenue = models.BigIntegerField(null=True, blank=True)
  ebit = models.BigIntegerField(null=True, blank=True)
  operating_income = models.BigIntegerField(null=True, blank=True)
  interest_expense = models.BigIntegerField(null=True, blank=True)
  earnings_before_tax = models.BigIntegerField(null=True, blank=True)
  operating_cashflow = models.BigIntegerField(null=True, blank=True)
  investment_cashflow = models.BigIntegerField(null=True, blank=True)
  total_assets = models.BigIntegerField(null=True, blank=True)
  total_liabilities = models.BigIntegerField(null=True, blank=True)
  data_type = models.CharField(max_length=100, blank=True, null=True)
  period = models.CharField(max_length=100, blank=True, null=True)
  class Meta:
    unique_together = ('stock', 'date')
  
  def __str__(self):
    return f"Financials for {self.stock.symbol} {self.date}"

class Sustainability(models.Model):
  stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='sustainability')
  date = models.DateField()
  esg_score = models.FloatField()
  carbon_footprint = models.FloatField()

  def __str__(self):
    return f"Sustainability for {self.stock.symbol} on {self.date}"
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