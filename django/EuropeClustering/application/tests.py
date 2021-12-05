from django.test import TestCase


# Create your tests here.
from application.models import Data
from functions import *


class MyTest(TestCase):

    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6)

    def test_kmeans_clustering(self):
        self.assertIsInstance(kmeans_clustering(data, 4), TimeSeriesKMeans)

    @classmethod
    def main(cls):
        # data = pd.DataFrame(list(Data.objects.all().values()))
        # data = data.drop('id', axis=1)
        # cls.test_kmeans_clustering(data, 4)
        cls.test_sum()
        print('All passed')


if _name_ == '_main_':
    MyTest.main()