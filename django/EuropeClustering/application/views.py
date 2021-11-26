from django.shortcuts import render

# Create your views here.


def homepage(request):
    return render(request, 'application/homepage.html')


def readabout(request):
    return render(request, 'application/readabout.html')