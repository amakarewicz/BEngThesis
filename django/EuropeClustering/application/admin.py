from django.contrib import admin

# Register your models here.
from .models import Variable, Algorithm

admin.site.register(Variable)
admin.site.register(Algorithm)