from django.shortcuts import render

# Create your views here.
from application.models import Variable, Algorithm, Data

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
from dtaidistance import dtw_ndim
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

# Using R inside python
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri

rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()

# Install packages
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)

# utils.install_packages('clValid')
# utils.install_packages('symbolicDA')

# Load packages
clValid = importr('clValid')
symbolicDA = importr('symbolicDA')

import warnings

warnings.filterwarnings(action='ignore')

# %matplotlib inline
plt.rcParams['figure.figsize'] = (9, 6)

pd.options.display.max_columns = None
pd.options.display.max_colwidth = None


def homepage(request):
    data = pd.DataFrame(list(Data.objects.all().values()))
    countries = data[['countrycode', 'country']].drop_duplicates().reset_index(drop=True)

    def kmeans_clustering(data: pd.DataFrame, n_clusters: int) -> TimeSeriesKMeans:
        """
        Perform KMeans clustering.

        Args:
            data (pd.DataFrame): preprocessed dataframe with economic indexes
            n_clusters (int): number of clusters to be formed

        Returns:
            TimeSeriesKMeans: fitted clustering model
        """
        # transform input data into adequate structure - 3D numpy array
        data_agg = data.drop('year', axis=1).groupby(['countrycode', 'country']).agg(list)
        n_countries = data_agg.shape[0]  # number of points (countries)
        time_range = len(data['year'].drop_duplicates())  # time range
        n_vars = data.shape[1] - 3  # number of economic indexes
        # filling the array
        data_agg_arr = np.empty(shape=(n_countries, n_vars, time_range))
        for i in range(data_agg.shape[0]):
            for j in range(data_agg.shape[1]):
                data_agg_arr[i][j] = np.array(data_agg.iloc[i, j])
        # creating and fitting a model
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw')
        model.fit(data_agg_arr)
        return model

    def plot_clustering(countries: pd.DataFrame, labels: np.array) -> None:
        """
        Plot cartogram presenting clustering results for given countries.

        Args:
            countries (pd.DataFrame): Pandas Dataframe containing at least one column, named 'countrycode',
            with ISO-3166 alpha-3 codes of countries
            labels (np.array): cluster assignment generated by clustering model for given countries
        """
        labels = labels.astype(str)
        countries["cluster"] = pd.Series(labels)
        fig = px.choropleth(countries, locations='countrycode', color="cluster",
                            projection='conic conformal', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_geos(lataxis_range=[35, 75], lonaxis_range=[-15, 45])  # customized to show Europe only
        return fig.to_html(full_html=False, default_height=500, default_width=700)

    model = kmeans_clustering(data, 4)
    figure = plot_clustering(countries, model.labels_)

    context = {'figure': figure}

    return render(request, 'application/homepage.html', context)


def readabout(request):
    variables = Variable.objects.all()
    algorithms = Algorithm.objects.all()

    context = {'variables': variables,
               'algorithms': algorithms}

    return render(request, 'application/readabout.html', context)
