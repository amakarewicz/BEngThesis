from django.test import TestCase


# Create your tests here.
class MyTest(TestCase):

    def test_sum(self):
        self.assertEqual(sum([1, 2, 2]), 6)

    @classmethod
    def main(cls):
        cls.test_sum()
        print('All passed')


if __name__ == '__main__':
    MyTest.main()
