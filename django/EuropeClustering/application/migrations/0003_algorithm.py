# Generated by Django 3.2.9 on 2021-11-26 12:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('application', '0002_rename_variables_variable'),
    ]

    operations = [
        migrations.CreateModel(
            name='Algorithm',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=30, verbose_name='Name')),
                ('description', models.CharField(max_length=800, verbose_name='Description')),
            ],
        ),
    ]
