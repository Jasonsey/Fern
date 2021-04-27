# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""data split"""
from typing import *

import pandas as pd
from sklearn.model_selection import train_test_split


def train_data_split(
        data: pd.DataFrame,
        test_size: Union[int, float],
        random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    使用sklearn的方法分割数据集

    Args:
        data: 待分割数据
        test_size: 如果是int，那么test数据是具体数量；如果是float，那么test数据会是比例
        random_state: 随机种子

    Returns:
        (train_df, test_df)
    """
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_df, test_df
