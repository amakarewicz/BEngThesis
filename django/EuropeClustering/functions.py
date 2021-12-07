import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import io
import urllib
import base64
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
    try:
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

    except Exception as ex:
        print(ex)


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

    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters, affinity='precomputed', linkage=linkage, compute_distances=True)
        model.fit(dtw_matrix)


    except Exception as ex:
        print(ex)
        model = AgglomerativeClustering(
            n_clusters=len(dtw_matrix), affinity='precomputed', linkage=linkage, compute_distances=True)
        model.fit(dtw_matrix)

    finally:
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


def plot_clustering(countries: pd.DataFrame, labels: np.array) -> str:
    """
    Plot cartogram presenting clustering results for given countries.

    Args:
        countries (pd.DataFrame): Pandas Dataframe containing at least one column, named 'countrycode',
        with ISO-3166 alpha-3 codes of countries
        labels (np.array): cluster assignment generated by clustering model for given countries
    """
    try:
        labels = labels.astype(str)
        countries["cluster"] = pd.Series(labels)
        fig = px.choropleth(countries, locations='countrycode', color="cluster",
                            projection='conic conformal', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_geos(lataxis_range=[35, 75], lonaxis_range=[-15, 45])  # customized to show Europe only
        fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)')
        return fig.to_html(full_html=False, default_height=400, default_width=400)

    except Exception as ex:
        print(ex)
        return str(ex)


def plot_dendrogram(model: AgglomerativeClustering, labels: np.array = None) -> None:
    """
    Create linkage matrix and plot dendrogram for given Hierarchical Clustering model.

    Args:
        model (AgglomerativeClustering): [description]
        labels (np.array): [description]
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
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    return uri


def plot_series(data: pd.DataFrame) -> str:
    """
    Plot time series of each variable, for each country.

    Args:
        data (pd.DataFrame): preprocessed dataframe with economic indexes
    """
    fig = go.Figure()
    buttons = list()
    for i in range(data.shape[1] - 3):
        ind = data.columns[i + 3,]
        df_test = data[['countrycode', 'year', ind]]
        # transposing
        df_test_transposed = df_test.pivot_table(index='countrycode', columns=['year'], values=ind).reset_index()
        df_test_final = df_test_transposed.rename_axis('').rename_axis("", axis="columns"
                                                                       ).set_index('countrycode')
        # Add Traces
        for countrycode in df_test_final.index:
            if i == 0:
                fig.add_trace(go.Scatter(x=df_test_final.columns, y=df_test_final.loc[countrycode],
                                         name=countrycode, visible=True))
            else:
                fig.add_trace(go.Scatter(x=df_test_final.columns, y=df_test_final.loc[countrycode],
                                         name=countrycode, visible=False))
        n_of_countries = df_test_final.shape[0]
        visible = [False] * n_of_countries * i + [True] * n_of_countries + [False] * n_of_countries * (
                n_of_countries - i - 1)
        buttons.append(dict(label=ind, method='update', args=[{'visible': visible}, {'title': ind}]))

    updatemenus = list([dict(active=0, buttons=buttons, xanchor='right', x=1, y=1.15)])
    fig.update_layout(updatemenus=updatemenus, title='Series',
                      title_x=0, title_xref='paper', margin=dict(l=20, r=20, t=20, b=20))

    return fig.to_html(full_html=False, default_height=400, default_width=500)


def evaluate_clustering(data: pd.DataFrame, labels: np.array) -> pd.DataFrame:
    """
    Evaluate used algorithm using following metrics: silhouette score, calinski-harabasz score, dunn index.

    Args:
        data (pd.DataFrame): preprocessed dataframe with economic indexes
        labels (np.array): cluster assignment generated by chosen clustering algorithm

    Returns:
        pd.DataFrame: metrics values
    """
    try:
        # transform input data into adequate structure - 3D numpy array
        data_t = data.melt(id_vars=['countrycode', 'country', 'year'])
        data_t = data_t.groupby(['countrycode', 'country', 'year', 'variable']
                                )['value'].aggregate('mean').unstack('year')
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
        # results = pd.DataFrame([sil_score, dunn_index, ch_score], columns=['Values'],
        #                        index=['Silhouette score', 'Dunn Index', 'Calinski-Harabasz score'])
        results = pd.DataFrame({'Index name': ['Silhouette score', 'Dunn Index', 'Calinski-Harabasz score'],
                                'Value': [sil_score, dunn_index, ch_score]})
    except Exception as ex:
        print(ex)
        results = pd.DataFrame({'Results': ['Application can not display results for one cluster.']})

    finally:
        return results


def plot_metrics(data: pd.DataFrame) -> None:
    """AI is creating summary for plot_metrics

    Args:
        data (pd.DataFrame): [description]
    """
    fig = go.Figure()
    buttons = list()
    for i in range(data.shape[1]-2):
        m = data.columns[i+2,]
        df_test = data[['algorithm','n_clusters', m]]

        # transposing
        df_test_transposed = df_test.pivot_table(index='algorithm', columns=['n_clusters'], values=m).reset_index()
        df_test_final = df_test_transposed.rename_axis('').rename_axis("", axis="columns").set_index('algorithm')

        # Add Traces
        for alg in df_test_final.index:
            if i==0:
                fig.add_trace(go.Scatter(x=df_test_final.columns, y=df_test_final.loc[alg],
                        name=alg, visible=True))            
            else:
                fig.add_trace(go.Scatter(x=df_test_final.columns, y=df_test_final.loc[alg],
                        name=alg, visible=False))
        n_of_countries = df_test_final.shape[0]
        visible = [False]*n_of_countries*i + [True]*n_of_countries + [False]*n_of_countries*(n_of_countries-i-1)
        buttons.append(dict(label = m,
                    method = 'update',
                    args = [{'visible': visible},
                            {'title': m}]))    
    fig.update_layout(dict(updatemenus=[
                        dict(type='dropdown', buttons=buttons, xanchor='right', x=1, y=1.15, active=0)#, showactive=True)                    ]
    ]))
    fig.update_layout(title='Metrics',title_x=0, title_xref='paper', margin=dict(l=20, r=20, t=20, b=20))
    return fig.to_html(full_html=False, default_height=400, default_width=500)


def plot_dbscan(data):
    """AI is creating summary for plot_dbscan

    Args:
        data ([type]): [description]
    """
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

    countries = data[['countrycode','country']].drop_duplicates().reset_index(drop=True)

    eps_grid = [3, 3.1, 3.2, 3.3, 3.4, 3.5]
    min_samples_grid = [2,3,4,5,6]
    plot_data = []
    for eps in eps_grid:
        for min_samples in min_samples_grid:
            model = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            model.fit(dtw_matrix)
            plot_data.append(dict(type='choropleth',
                    locations = countries['countrycode'].astype(str),
                    z=model.labels_, showscale=False))
    # (3,2), (3,3) ... (3.1, 2), (3.1, 3) 

    steps = []
    i=0
    for eps in eps_grid:
        for min_samples in min_samples_grid:
            step = dict(method='restyle',
                        args=['visible', [False] * len(plot_data)],
                        label='{} / {}'.format(eps, min_samples))
            step['args'][1][i] = True
            steps.append(step)
            i+=1

    sliders = [dict(active=0,
                    pad={"t": 1},
                    steps=steps,
                    currentvalue={'prefix' : 'Eps - ', 'suffix' : ' - min samples'})]
    
    layout = dict(geo=dict(projection={'type': 'conic conformal'}, lataxis={'range':[35,75]},
                           lonaxis={'range': [-15, 45]}), sliders=sliders)
    fig = go.Figure(dict(data=plot_data, layout=layout))
    fig.update_layout(showlegend=False)
    fig.update_traces(showlegend=False, selector = dict(type='choropleth'))
    return fig.to_html(full_html=False, default_height=400, default_width=500)