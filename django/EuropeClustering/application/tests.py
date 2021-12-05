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

        self.assertIsInstance(kmeans_clustering(data, 2), TimeSeriesKMeans)

    def agglomerative_clustering(self):
        data = pd.DataFrame({'countrycode': ['ALB', 'ALB', 'AUT', 'AUT', 'ROU', 'ROU'],
                             'country': ['Albania', 'Albania', 'Austria', 'Austria', 'Romania', 'Romania'],
                             'year': [1991, 1992, 1991, 1992, 1991, 1992],
                             'pop': [1234, 4234, 3452, 2342, 2342, 2354],
                             'rgdpna': [9856, 3485, 9823, 3847, 3485, 9232]})

        self.assertIsInstance(agglomerative_clustering(data, 2, 'complete'), TimeSeriesKMeans)

    def dbscan_clustering(self):
        data = pd.DataFrame({'countrycode': ['ALB', 'ALB', 'AUT', 'AUT', 'ROU', 'ROU'],
                             'country': ['Albania', 'Albania', 'Austria', 'Austria', 'Romania', 'Romania'],
                             'year': [1991, 1992, 1991, 1992, 1991, 1992],
                             'pop': [1234, 4234, 3452, 2342, 2342, 2354],
                             'rgdpna': [9856, 3485, 9823, 3847, 3485, 9232]})

        self.assertIsInstance(dbscan_clustering(data, 2.3, 4), TimeSeriesKMeans)

    @classmethod
    def main(cls):
        cls.test_kmeans_clustering()
        cls.agglomerative_clustering()
        cls.dbscan_clustering()
        print('All passed')


if __name__ == '__main__':
    MyTest.main()
