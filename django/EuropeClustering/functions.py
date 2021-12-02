import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
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


def agglomerative_clustering(data: pd.DataFrame, n_clusters: int, linkage: str) -> AgglomerativeClustering:
    """
    Perform hierarchical clustering.

    Args:
        data (pd.DataFrame): preprocessed dataframe with economic indexes
        n_clusters (int): number of clusters to be formed
        linkage (str): type of linkage criterion; 'average', 'complete' or 'single'

    Returns:
        AgglomerativeClustering: fitted clustering model
    """
    # transform input data into adequate structure - 3D numpy array
    data_t = data.melt(id_vars=['countrycode', 'country', 'year'])
    data_t = data_t.groupby(['countrycode', 'country', 'year', 'variable'])['value'].aggregate('mean').unstack(
        'year')
    data_t = data_t.reset_index().drop('variable', axis=1).groupby(['countrycode', 'country']).agg(list)
    n_countries = data_t.shape[0]  # number of points (countries)
    time_range = data_t.shape[1]  # time range
    n_vars = data.shape[1] - 3  # number of economic indexes
    # filling the array
    data_t_arr = np.empty(shape=(n_countries, time_range, n_vars))
    for i in range(n_countries):
        for j in range(time_range):
            data_t_arr[i][j] = np.array(data_t.iloc[i, j])
    # calculating distances between points (countries)
    dtw_matrix = dtw_ndim.distance_matrix_fast(data_t_arr, n_vars)
    # creating and fitting the model
    model = AgglomerativeClustering(
        n_clusters=n_clusters, affinity='precomputed', linkage=linkage, compute_distances=True)
    model.fit(dtw_matrix)
    return model


def dbscan_clustering(data: pd.DataFrame, eps: float, min_samples: int) -> DBSCAN:
    """
    Perform DBSCAN clustering.

    Args:
        data (pd.DataFrame): preprocessed dataframe with economic indexes
        eps (float): maximum distance between two points for them to be considered as neighbouring
        min_samples (int): number of samples in a neighborhood for a point to be considered as a core point

    Returns:
        DBSCAN: fitted clustering model
    """
    # transform input data into adequate structure - 3D numpy array
    data_t = data.melt(id_vars=['countrycode', 'country', 'year'])
    data_t = data_t.groupby(['countrycode', 'country', 'year', 'variable'])['value'].aggregate('mean').unstack(
        'year')
    data_t = data_t.reset_index().drop('variable', axis=1).groupby(['countrycode', 'country']).agg(list)
    n_countries = data_t.shape[0]  # number of points (countries)
    time_range = data_t.shape[1]  # time range
    n_vars = data.shape[1] - 3  # number of economic indexes
    # filling the array
    data_t_arr = np.empty(shape=(n_countries, time_range, n_vars))
    for i in range(n_countries):
        for j in range(time_range):
            data_t_arr[i][j] = np.array(data_t.iloc[i, j])
    # calculating distances between points (countries)
    dtw_matrix = dtw_ndim.distance_matrix_fast(data_t_arr, n_vars)
    # creating and fitting the model
    model = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    model.fit(dtw_matrix)
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


def plot_dendrogram(model: AgglomerativeClustering, labels: np.array = None) -> None:
    """
    Create linkage matrix and plot dendrogram for given Hierarchical Clustering model.

    Args:
        model (AgglomerativeClustering): [description]
    """
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    # create linkage matrix from calculated counts
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    # plot the  dendrogram
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(1, 1, 1)
    dendrogram(linkage_matrix, labels=labels)
    plt.xlabel('country codes', fontsize=20)
    plt.ylabel('distance', fontsize=20)
    ax.tick_params(axis='x', which='major', labelsize=20)
    ax.tick_params(axis='y', which='major', labelsize=20)
    return fig.to_html(full_html=False, default_height=500, default_width=700)


def evaluate_clustering(data: pd.DataFrame, labels: np.array) -> pd.DataFrame:
    """
    Evaluate used algorithm using following metrics: silhouette score, calinski-harabasz score, dunn index.

    Args:
        data (pd.DataFrame): preprocessed dataframe with economic indexes
        labels (np.array): cluster assignment generated by chosen clustering algorithm

    Returns:
        pd.DataFrame: metrics values
    """
    # transform input data into adequate structure - 3D numpy array
    data_t = data.melt(id_vars=['countrycode', 'country', 'year'])
    data_t = data_t.groupby(['countrycode', 'country', 'year', 'variable'])['value'].aggregate('mean').unstack(
        'year')
    data_t = data_t.reset_index().drop('variable', axis=1).groupby(['countrycode', 'country']).agg(list)
    n_countries = data_t.shape[0]  # number of points (countries)
    time_range = data_t.shape[1]  # time range
    n_vars = data.shape[1] - 3  # number of economic indexes
    # filling the array
    data_t_arr = np.empty(shape=(n_countries, time_range, n_vars))
    for i in range(n_countries):
        for j in range(time_range):
            data_t_arr[i][j] = np.array(data_t.iloc[i, j])
    # calculating distances between points (countries)
    dtw_matrix = dtw_ndim.distance_matrix_fast(data_t_arr, n_vars)
    # calculating metric values
    sil_score = silhouette_score(dtw_matrix, labels, metric='precomputed')
    dunn_index = clValid.dunn(dtw_matrix, labels)[0]
    ch_score = symbolicDA.index_G1d(dtw_matrix, labels)[0]
    results = pd.DataFrame([sil_score, dunn_index, ch_score], columns=['Values'],
                           index=['Silhouette score', 'Dunn Index', 'Calinski-Harabasz score'])
    return results
