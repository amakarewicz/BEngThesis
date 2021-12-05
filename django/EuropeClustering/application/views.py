from django.http import HttpResponseRedirect
from django.shortcuts import render
from functions import *

# Create your views here.
from application.forms import CustomizeReport
from application.models import Variable, Algorithm, Data

import pandas as pd


def homepage(request):
    data = pd.DataFrame(list(Data.objects.all().values()))
    data = data.drop('id', axis=1)
    countries = data[['countrycode', 'country']].drop_duplicates().reset_index(drop=True)

    if request.method == 'POST':
        form = CustomizeReport(request.POST)
        if form.is_valid():
            variables = form.cleaned_data['variables']
            columns = ['country', 'countrycode', 'year'] + list(variables.values_list('name', flat=True))
            data = data[columns]
            if form.cleaned_data['algorithm'] == 'kmeans':
                model = kmeans_clustering(data, int(form.cleaned_data['n_clusters']))
                other_graph = '0'
            elif form.cleaned_data['algorithm'] == 'hierarchical':
                model = agglomerative_clustering(data, int(form.cleaned_data['n_clusters']), form.cleaned_data['linkage'])
                other_graph = plot_dendrogram(model, model.labels_)
            else:
                model = dbscan_clustering(data, int(form.cleaned_data['eps']), int(form.cleaned_data['min_samples']))
                other_graph = '0'
            figure = plot_clustering(countries, model.labels_)
            table = evaluate_clustering(data, model.labels_).to_html()
            series = plot_series(data)
            context = {'figure': figure,
                       'table': table,
                       'form': form,
                       'series': series,
                       'other_graph': other_graph}
            return render(request, 'application/homepage.html', context)
        else:
            form = CustomizeReport()
            context = {'form': form}

    else:
        form = CustomizeReport()
        context = {'form': form}

    return render(request, 'application/homepage.html', context)


def readabout(request):
    variables = Variable.objects.all()
    algorithms = Algorithm.objects.all()

    context = {'variables': variables,
               'algorithms': algorithms}

    return render(request, 'application/readabout.html', context)
