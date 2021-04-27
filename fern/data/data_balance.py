# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""data balance"""
from typing import *
from copy import deepcopy

import numpy as np
import pandas as pd


def over_sample(data: pd.DataFrame, label_col: str, rate: float, random_state: Optional[int] = None) -> pd.DataFrame:
    """
    过采样以最大值为基准：
        1. 样本数低于比例的都会被over sample
        2. 样本数高于于比例的不变

    支持的标签类型：
        1. 多任务: {'task1': array([1, 1, 0], 'task2': array([1, 0, 0]}
        2. 单任务多标签: array([1, 1, 0])
        3. 单任务单标签: array([1, 0, 0])

    Args:
        data: 需要上采样的数据
        label_col: 需要处理的标签列名
        rate: over sample之后，所有的类别的数量不小于 rate x 最大类别样本数
        random_state: 随机种子

    Returns:
        采样之后的列数据
    """
    res_data = deepcopy(data)
    label_example = data[label_col].iloc[0]
    if isinstance(label_example, dict):
        key = list(label_example.keys())[0]

    def generate_label_evaluate_standard(item):
        if isinstance(item, dict):
            item = item[key]
        return item

    labels = np.vstack(data[label_col].map(generate_label_evaluate_standard))
    count_label = np.sum(labels, axis=0)
    target_num = int(max(count_label) * rate)
    # 遍历所有的样本数量，并随机采样到至少target num
    for col_idx, num in sorted(enumerate(count_label), key=lambda item: item[1]):
        row_idx = np.argwhere(labels[:, col_idx] == 1).reshape(-1)
        row_idx = np.random.RandomState(random_state).permutation(row_idx)
        if num >= target_num or num < 2:
            continue
        repeat_times = int(target_num // num - 1)
        copy_num = int(target_num % num)

        _data = data.iloc[row_idx]
        res_data.append(_data.loc[_data.index.repeat(repeat_times)])
        res_data.append(_data.iloc[:copy_num])

    random_idx = np.random.RandomState(random_state).permutation(res_data.shape[0])
    res_data = res_data.iloc[random_idx]
    return res_data
