# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""test data tools"""
import numpy as np

from fern.data import FernSeries, FernDataFrame


class TestFernSeries(object):
    @classmethod
    def setup_class(cls):
        cls.data_0 = np.arange(10, 20)
        cls.data_1 = list('abcdefghij')
        cls.idx = np.random.randint(10, size=(10,))

    def test_parallel_map(self):
        data_0 = FernSeries(data=self.data_0, index=self.idx)
        data_1 = FernSeries(data=self.data_1, index=self.idx)

        res_0_pred = data_0.parallel_map(self.square, processes=2)
        res_0_true = data_0.map(self.square)
        assert np.all(res_0_pred == res_0_true)

        res_1_pred = data_0.parallel_map(self.multiply, args=(3,), processes=2)
        res_1_true = data_0.map(lambda x: self.multiply(x, 3))
        assert np.all(res_1_pred == res_1_true)

        try:
            data_1.parallel_map(self.square, processes=2)
        except TypeError as err:
            pass

    @staticmethod
    def square(x):
        return x ** 2

    @staticmethod
    def multiply(x, y):
        return x * y


class TestFernDataFrame(object):
    @classmethod
    def setup_class(cls):
        cls.data = np.arange(10, 20)
        cls.idx = np.random.randint(10, size=(10,))

    def test_parallel_apply(self):
        data = FernSeries(data=self.data, index=self.idx)
        df = FernDataFrame({'d': data})

        res_pred = df.parallel_apply(self.square_dict, axis=1, processes=2)
        res_true = data.map(self.square)
        assert np.all(res_pred == res_true)

    @staticmethod
    def square_dict(x: dict):
        return x['d'] ** 2

    @staticmethod
    def square(x):
        return x ** 2
