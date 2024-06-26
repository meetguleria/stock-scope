# Generated by Django 4.2.11 on 2024-05-03 12:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stocks', '0004_financials_data_type_financials_period'),
    ]

    operations = [
        migrations.AlterField(
            model_name='financials',
            name='earnings_before_tax',
            field=models.BigIntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='financials',
            name='ebit',
            field=models.BigIntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='financials',
            name='interest_expense',
            field=models.BigIntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='financials',
            name='investment_cashflow',
            field=models.BigIntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='financials',
            name='net_income',
            field=models.BigIntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='financials',
            name='operating_cashflow',
            field=models.BigIntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='financials',
            name='operating_income',
            field=models.BigIntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='financials',
            name='total_assets',
            field=models.BigIntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='financials',
            name='total_liabilities',
            field=models.BigIntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='financials',
            name='total_revenue',
            field=models.BigIntegerField(default=0),
        ),
    ]
