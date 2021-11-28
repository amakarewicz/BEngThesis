from django.contrib import admin

# Register your models here.
from .models import Variable, Algorithm, Data

admin.site.register(Variable)
admin.site.register(Algorithm)
admin.site.register(Data)
