from django.shortcuts import render

# Create your views here.
from application.models import Variable


def homepage(request):
    return render(request, 'application/homepage.html')


def readabout(request):
    variables = Variable.objects.all()

    context = {'variables': variables}

    return render(request, 'application/readabout.html', context)
    # return render(request, 'application/readabout.html')