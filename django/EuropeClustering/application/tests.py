from django.test import TestCase

# Create your tests here.
from application.models import Data
from functions import *


class MyTest(TestCase):

    def test_kmeans_clustering(self):
        data = pd.DataFrame({'countrycode': ['ALB', 'ALB', 'AUT', 'AUT', 'ROU', 'ROU'],
                             'country': ['Albania', 'Albania', 'Austria', 'Austria', 'Romania', 'Romania'],
                             'year': [1991, 1992, 1991, 1992, 1991, 1992],
                             'pop': [1234, 4234, 3452, 2342, 2342, 2354],
                             'rgdpna': [9856, 3485, 9823, 3847, 3485, 9232]})

        for k in range(2, 10, 1):
            self.assertIsInstance(kmeans_clustering(data, k), TimeSeriesKMeans)

    def test_agglomerative_clustering(self):
        data = pd.DataFrame({'countrycode': ['ALB', 'ALB', 'AUT', 'AUT', 'ROU', 'ROU'],
                             'country': ['Albania', 'Albania', 'Austria', 'Austria', 'Romania', 'Romania'],
                             'year': [1991, 1992, 1991, 1992, 1991, 1992],
                             'pop': [1234, 4234, 3452, 2342, 2342, 2354],
                             'rgdpna': [9856, 3485, 9823, 3847, 3485, 9232]})

        for k in range(2, 10, 1):
            self.assertIsInstance(agglomerative_clustering(data, k, 'complete'), AgglomerativeClustering)
            self.assertIsInstance(agglomerative_clustering(data, k, 'single'), AgglomerativeClustering)
            self.assertIsInstance(agglomerative_clustering(data, k, 'average'), AgglomerativeClustering)

    def test_dbscan_clustering(self):
        data = pd.DataFrame({'countrycode': ['ALB', 'ALB', 'AUT', 'AUT', 'ROU', 'ROU'],
                             'country': ['Albania', 'Albania', 'Austria', 'Austria', 'Romania', 'Romania'],
                             'year': [1991, 1992, 1991, 1992, 1991, 1992],
                             'pop': [1234, 4234, 3452, 2342, 2342, 2354],
                             'rgdpna': [9856, 3485, 9823, 3847, 3485, 9232]})

        for eps in np.arange(2, 10, 0.1):
            for min_samples in range(2, 19, 1):
                self.assertIsInstance(dbscan_clustering(data, eps, min_samples), DBSCAN)

    def test_plot_clustering(self):
        countries = pd.DataFrame({'countrycode': ['ALB', 'AUT', 'ROU', 'MNE', 'POL', 'MDA', 'ISL'],
                             'country': ['Albania', 'Austria', 'Romania', 'Montenegro', 'Poland', 'Macedonia', 'Iceland']})

        self.assertIsInstance(plot_clustering(countries, np.array([0, 1, 1, 2, 3, 2, 1])), str)
        self.assertIsInstance(plot_clustering(countries, np.array([0, 1, 1, 2, 3, 2])), str)
        self.assertIsInstance(plot_clustering(countries, np.array([0, 1, 2, 3, 4, 5, 7])), str)

    def test_plot_series(self):
        data = pd.DataFrame({'countrycode': ['ALB', 'ALB', 'AUT', 'AUT', 'ROU', 'ROU'],
                             'country': ['Albania', 'Albania', 'Austria', 'Austria', 'Romania', 'Romania'],
                             'year': [1991, 1992, 1991, 1992, 1991, 1992],
                             'pop': [1234, 4234, 3452, 2342, 2342, 2354],
                             'rgdpna': [9856, 3485, 9823, 3847, 3485, 9232]})

        self.assertIsInstance(plot_series(data), str)

    def test_evaluate_clustering(self):
        data = pd.DataFrame({'countrycode': ['ALB', 'ALB', 'AUT', 'AUT', 'ROU', 'ROU'],
                             'country': ['Albania', 'Albania', 'Austria', 'Austria', 'Romania', 'Romania'],
                             'year': [1991, 1992, 1991, 1992, 1991, 1992],
                             'pop': [1234, 4234, 3452, 2342, 2342, 2354],
                             'rgdpna': [9856, 3485, 9823, 3847, 3485, 9232]})

        self.assertIsInstance(evaluate_clustering(data, np.array([0, 1, 2])), pd.DataFrame)
        self.assertIsInstance(evaluate_clustering(data, np.array([0, 0, 0])), pd.DataFrame)



    @classmethod
    def main(cls):
        cls.test_kmeans_clustering()
        cls.test_agglomerative_clustering()
        cls.test_dbscan_clustering()
        cls.test_plot_clustering()
        cls.test_plot_series()
        cls.test_evaluate_clustering()
        print('All passed')


if __name__ == '__main__':
    MyTest.main()
