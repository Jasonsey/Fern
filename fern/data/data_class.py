# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""data classes"""
from typing import *
import csv
from multiprocessing.pool import Pool

import pandas as pd
from fern.utils import check_path


class FernSeries(pd.Series):
    @property
    def _constructor(self):
        return FernSeries

    @property
    def _constructor_expanddim(self):
        return FernDataFrame

    def parallel_map(self, func: Callable, args: Optional[Tuple] = None, processes: Optional[int] = None):
        """
        map method that use multiple processes

        Args:
            func: called to preprocess data
            args: args to be fed into the func, in addition to the original data
            processes: how many cpu to be used, default use all cpu

        Returns:
            processed series

        Examples:
            >>> d = FernSeries([1,2,3,4])
            >>> def multiply(x, y):
            ...     return x * y
            >>> d.parallel_map(func=multiply, args=(2, ))
            0    2
            1    4
            2    6
            3    8
            dtype: int64
        """
        args = args if args is not None else tuple()
        with Pool(processes) as pool:
            multi_res = [pool.apply_async(func, (item, *args)) for item in self.array]
            new_array = [data.get() for data in multi_res]
        return FernSeries(data=new_array, index=self.index)


class FernDataFrame(pd.DataFrame):
    """sub class of pandas.DataFrame with additional function"""

    @property
    def _constructor_expanddim(self):
        """for lint check"""
        raise NotImplementedError

    @property
    def _constructor(self):
        return FernDataFrame

    @property
    def _constructor_sliced(self):
        return FernSeries

    def parallel_apply(
        self,
        func: Callable[[Dict], Any],
        args: Optional[Tuple] = None,
        axis: int = 1,
        processes: Optional[int] = None
    ) -> FernSeries:
        """
        apply method which uses multiple processes

        Args:
            func: Function to apply to each column or row.
            args: args to be fed into the func, in addition to the original data
            axis: Axis along which the function is applied
                * 0: apply function to each column.
                * 1: apply function to each row.
            processes: how many cpu to be used, default use all cpu

        Returns:
            processed series
        """
        data: FernSeries = self.apply(lambda x: x.to_dict(), axis=axis)
        res = data.parallel_map(func=func, args=args, processes=processes)
        return res

    def save(self, path):
        """
        save data frame with csv format to path

        Parameters
        ----------
        path : str, Path
            path where data save
        """
        check_path(path)
        self.to_csv(path, index=True, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
