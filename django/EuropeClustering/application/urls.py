from django.urls import path
from . import views


urlpatterns = [
    path('homepage/', views.homepage, name='homepage'),
    path('readabout/', views.readabout, name='readabout'),
    path('report/', views.report, name='report')
]