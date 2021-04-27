# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
from typing import *

import pandas as pd
from sqlalchemy import create_engine

from fern.data import FernDataFrame


def read_sql(url: str, sql: str, index: Optional[str] = None, drop: bool = False) -> FernDataFrame:
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
    engine = create_engine(url=url)

    df = pd.read_sql(sql, engine)
    df = FernDataFrame(df)

    if drop:
        df = df.dropna()
    if index is None:
        data = df
    else:
        data = df.set_index(index)
    return data
