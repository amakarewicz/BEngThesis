from django.contrib import admin

# Register your models here.
from .models import Variable, Algorithm, Data, MetricsValues, DataOriginal

admin.site.register(Variable)
admin.site.register(Algorithm)
admin.site.register(Data)
admin.site.register(MetricsValues)
admin.site.register(DataOriginal)
