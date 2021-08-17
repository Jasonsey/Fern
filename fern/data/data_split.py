# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""data split"""
from typing import *

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_data_split(
        data: pd.DataFrame,
        test_size: Union[int, float],
        max_test_size: Optional[int] = None,
        random_state: Union[int, np.random.RandomState, None] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    使用sklearn的方法分割数据集

    Args:
        data: 待分割数据
        test_size: 如果是int，那么test数据是具体数量；如果是float，那么test数据会是比例
        max_test_size: 如果test_size是float, 那么可以使用这个字段限制最大的测试集大小
        random_state: 随机种子

    Returns:
        (train_df, test_df)
    """
    if isinstance(test_size, float) and test_size <= 1:
        test_size = int(len(data) * test_size)
        if isinstance(max_test_size, int) and test_size > max_test_size:
            test_size = max_test_size
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_df, test_df
