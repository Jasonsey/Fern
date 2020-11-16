# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""test data tools"""
import numpy as np

from fern.utils.data import FernSeries, FernDataFrame


def square(x):
    return x ** 2


class TestFernSeries(object):
    @classmethod
    def setup_class(cls):
        cls.data_0 = np.arange(10, 20)
        cls.data_1 = list('abcdefghij')
        cls.idx = np.random.randint(10, size=(10,))

    def test_parallel_map(self):
        data_0 = FernSeries(data=self.data_0, index=self.idx)
        data_1 = FernSeries(data=self.data_1, index=self.idx)

        res_0_pred = data_0.parallel_map(square, 2)
        res_0_true = data_0.map(square)
        assert np.all(res_0_pred == res_0_true)

        try:
            data_1.parallel_map(square, 2)
        except TypeError as err:
            pass


class TestFernDataFrame(object):
    @classmethod
    def setup_class(cls):
        cls.data = np.arange(10, 20)
        cls.idx = np.random.randint(10, size=(10,))

    def test_parallel_map(self):
        data = FernSeries(data=self.data, index=self.idx)
        df = FernDataFrame({'d': data})

        res_pred = df['d'].parallel_map(square, 2)
        res_true = data.map(square)
        assert np.all(res_pred == res_true)
