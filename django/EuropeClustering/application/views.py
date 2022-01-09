from django.http import HttpResponseRedirect
from django.shortcuts import render
from functions import *

# Create your views here.
from application.forms import CustomizeReport
from application.models import *

import pandas as pd


def homepage(request):
    data = pd.DataFrame(list(Data.objects.all().values()))
    data = data.drop('id', axis=1)
    countries = data[['countrycode', 'country']].drop_duplicates().reset_index(drop=True)

    dataOriginal = pd.DataFrame(list(DataOriginal.objects.all().values()))
    dataOriginal = dataOriginal.drop('id', axis=1)

    if request.method == 'POST':
        form = CustomizeReport(request.POST)
        if form.is_valid():
            variables = form.cleaned_data['variables']
            columns = ['country', 'countrycode', 'year'] + list(variables.values_list('name', flat=True))
            data = data[columns]
            if form.cleaned_data['algorithm'] == 'kmeans':
                model = kmeans_clustering(data, int(form.cleaned_data['n_clusters']))
            elif form.cleaned_data['algorithm'] == 'hierarchical':
                model = agglomerative_clustering(data, int(form.cleaned_data['n_clusters']),
                                                 form.cleaned_data['linkage'])
                #other_graph = plot_dendrogram(model, np.array(countries.countrycode))
            else:
                model = dbscan_clustering(data, float(form.cleaned_data['eps']), int(form.cleaned_data['min_samples']))

            colors = ['#ffe8d6', '#ddbea9', '#cb997e', '#b7b7a4', '#a5a58d', '#6b705c']
            colors = colors[:(max(model.labels_) + 1)]

            figure = plot_clustering(countries, model.labels_, colors)
            eval_clustering = evaluate_clustering(data, model.labels_)
            table = eval_clustering.to_html(index=False).\
                replace('<thead>', '<thead id="tbody">').replace("</thead>", "").replace("<tbody>", "").\
                replace('<thead id="tbody">', '<tbody>')
            cluster_info = print_cluster_info(dataOriginal, model.labels_).to_html(index=False).\
                replace('<thead>', '<thead id="tbody">').replace("</thead>", "").replace("<tbody>", "").\
                replace('<thead id="tbody">', '<tbody>')
            series = plot_series(dataOriginal)
            if eval_clustering.shape[0] > 1:
                context = {'figure': figure,
                           'table': table,
                           'form': form,
                           'series': series,
                           'cluster_info': cluster_info,
                           'countries': countries['countrycode'].astype(str),
                           'labels': model.labels_}
            else:
                context = {
                    'table': table,
                    'form': form
                }
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

    context = {'variables': variables}

    return render(request, 'application/readabout.html', context)


def report(request):
    data = pd.DataFrame(list(Data.objects.all().values()))
    data = data.drop('id', axis=1)
    countries = data[['countrycode', 'country']].drop_duplicates().reset_index(drop=True)
    metric_data = pd.DataFrame(list(MetricsValues.objects.all().values()))
    metric_data = metric_data.drop('id', axis=1)
    metrics = plot_metrics(metric_data)
    dbscan = plot_dbscan(data)
    context = {'metrics': metrics, 'dbscan': dbscan}
    return render(request, 'application/report.html', context)
