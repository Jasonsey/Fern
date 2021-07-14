# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""pipelines tools"""
from typing import *

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from fern.data import FernDataFrame


class FernDownloader(object):
    def read_sql(self, url: str, sql: str, index: Optional[str] = None, drop: bool = False) -> FernDataFrame:
        """
        通过sql, 从数据库读取数据

        Args:
            url: 用于连接数据库, 例如: 'mysql+pymysql://user:passwd@hostname/dbname?charset=utf8mb4'
            sql: 读取数据库的sql语句
            index: 如果提供, 会根据这个字段, 创建返回的dataframe的index列
            drop: 默认不移除空白行

        Returns:
            从数据库中下载的原始数据
        """
        raise NotImplementedError


class FernCleaner(object):
    """
    data cleaner
    """
    def clean(self, data: Union[pd.DataFrame, FernDataFrame]) -> FernDataFrame:
        """
        clean data main entry, 如果清洗数据过程中需要用到多进程，那么需要保证被调用函数是静态的

        一般流程：
            1. 清洗原始数据
            2. 清洗label数据

        Args:
            data: source data frame

        Returns:
            cleaned data with columns, data_col and label_col
        """
        raise NotImplementedError

    def clean_data(self, row: Union[pd.Series, dict]) -> Any:
        """
        convert input data function for pandas apply function

        due to the limitation of parallel process, the function need to be static method

        一般流程：
            1. 关键数据截取
            2. 特殊字符清理
            3. 停用词移除(不在此处操作, 这个在tokenize中操作)

        Args:
            row: the input data row

        Returns:
            清理好的文本数据。通常不同文本段这件拼接使用空格，这样在下一步操作的时候，这个空格就会被直接移除
        """
        raise NotImplementedError

    def clean_label(self, row: Union[pd.Series, dict]) -> Any:
        """
        clean label function for pandas apply function

        一般流程：
            1. 只做标签格式化操作，通常是大小写等简单操作

        注意事项：
            1. 如果不是必要可以不要做标签清理
            2. 如果是单任务单标签可以整理成 label1 的数据格式
            3. 如果是单任务多标签形式，那么可以整理成 [label1, label2]的数据格式
            4. 如果是多任务形式，那么可以整理成 {'task1': [label1, label2], 'task2': [label1, label2]}的数据格式

        Args:
            row: find source label data

        Returns:
            cleaned label
        """
        raise NotImplementedError


class FernTokenizer(object):
    """
    转换清洗完的数据为token列表格式，不涉及到pad和生成array的操作
    """
    def tokenize(self, data: Union[pd.DataFrame, FernDataFrame]) -> Union[pd.DataFrame, FernDataFrame]:
        """
        转化一个data col 和label col对应的数据为token格式，但是不做padding的操作

        一般操作流程：
            1. token data
            2. token label
            3. 保存数据

        Args:
            data: the data frame where the cleaned data is stored

        Returns:
            The transformed word id sequence
        """
        raise NotImplementedError

    def tokenize_data(self, string: str) -> np.ndarray:
        """
        转化string文本为token列表格式

        一般操作步骤：
            1. 文本分词为词列表
            2. 使用停用词库移除停用词
            3. 词库中未出现的词, 转化为ukn token
            2. 如果必要，可以添加特殊token到这个列表中
            2. 使用word2id转化为token列表

        Args:
            string: 待转化文本

        Returns:
            The transformed word id sequence
        """
        raise NotImplementedError

    def tokenize_label(self, label: Union[dict, list, tuple, str]) -> Union[dict, np.ndarray]:
        """
        转化label为id array格式

        Args:
            label: to be transform source label data

        Returns:
            如果是多任务，那么label格式是字典，此时返回数据也需要是字典；array对象
        """
        raise NotImplementedError


class FernSplitter(object):
    """split data into train data and val data"""
    def __init__(self, rate_val: float, random_state: Optional[int] = None):
        self.rate_val = rate_val
        self.random_state = random_state

    def split(self, data: Union[pd.DataFrame, FernDataFrame]) -> Union[pd.DataFrame, FernDataFrame]:
        """
        split function to split data

        一般流程:
            1. 分割数据集为训练和评估数据集 (一般测试数据集会在更早之前被分割出去)
            2. 保存数据集

        Args:
            data: 待分割数据集
        """
        raise NotImplementedError


class FernBalance(object):
    """
    Class for sample balance
    """
    def balance(self, data: Union[pd.DataFrame, FernDataFrame]) -> Union[pd.DataFrame, FernDataFrame]:
        """
        根据类别数量，平衡样本数量

        一般流程：
            1. 制作平衡好的数据集
            2. 保存数据

        Args:
            data: 需要平衡的数据集
        """
        raise NotImplementedError
