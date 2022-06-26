# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""data utils"""
from typing import *
import re
import pickle
import pathlib

import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
from pyarrow import csv
from smart_open import hdfs

from fern.utils import check_path, HDFSPath, is_hdfs_path
from fern.data import FernDataFrame


def save_to_csv(data: Union[pd.DataFrame, FernDataFrame], path: Union[str, pathlib.Path]):
    """
    save data to path

    Args:
        data: 需要保存的data frame
        path: path where data save
    """
    check_path(path)
    if not data:
        raise ValueError('You should get source data before save')
    data = FernDataFrame(data)
    data.save(path)


def read_csv(path: Union[str, pathlib.Path, HDFSPath], index_col: str = None, eval_col: Optional[List[str]] = None):
    """
    load data from path, support hdfs and local file path

    Args:
        path: path where data save
        index_col: 需要初始化的index 列
        eval_col: 需要恢复数据格式的数据列，读取的数据默认是string格式
    """
    if not is_hdfs_path(str(path)):
        data = pd.read_csv(path)
    else:
        path = HDFSPath(path)
        with hdfs.open(str(path), 'rb') as f:
            data = csv.read_csv(f).to_pandas()

    if index_col:
        data = data.set_index(index_col)
    if isinstance(eval_col, list):
        for col in eval_col:
            data.loc[:, col] = data[col].map(eval)
    return data


def read_orc(path: Union[str, pathlib.Path, HDFSPath], index_col: str = None, eval_col: Optional[List[str]] = None):
    """
    load data from path, support hdfs and local file path

    Args:
        path: path where data save
        index_col: 需要初始化的index 列
        eval_col: 需要恢复数据格式的数据列，读取的数据默认是string格式
    """
    customized_open = hdfs.open if 'hdfs' in str(path) else open
    with customized_open(str(path), 'rb') as file_io:
        byte_buffer = pa.BufferReader(file_io.read())
    orc_file = orc.ORCFile(byte_buffer)
    data = orc_file.read().to_pandas()

    if index_col:
        data = data.set_index(index_col)
    if isinstance(eval_col, list):
        for col in eval_col:
            data.loc[:, col] = data[col].map(eval)
    return data


def save_to_pickle(data: Any, path: Union[str, pathlib.Path]):
    """
    save data to path

    Args:
        data: 待保存的数据
        path: path where data save
    """
    check_path(path)
    if data is None:
        raise ValueError('You should get source data before save')
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=4)


def load_from_pickle(path: Union[str, pathlib.Path]):
    """
    load data from path

    Args:
        path: path where data save
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def read_words(words_path):
    """
    Read user words, stop words and word library from path

    Lines beginning with `#` or consisting entirely of white space characters will be ignored

    Parameters
    ----------
    words_path : str, Path, None
        words path

    Returns
    -------
    list[str]
        user word list and stop word list
    """

    def read(path):
        res = set()
        with open(path, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().lower()
                if line and line[0] != '#':
                    res.add(line)
        res = list(res)
        return res

    if words_path is None or not pathlib.Path(words_path).exists():
        words = []
    else:
        words = read(words_path)
    return words


def read_regex_words(words_path):
    """
    Read words written through the regex

    Parameters
    ----------
    words_path : str, Path, None
        words path

    Returns
    -------
    list[re.Pattern]
        user word list and stop word list
    """
    words = read_words(words_path)
    word_reg = [re.compile(word) for word in words]
    return word_reg


def read_library_size(path):
    """
    read the length of the word/label library
    this will skip the space line automatically

    Parameters
    ----------
    path : str, pathlib.Path
        word library path

    Returns
    -------
    int
        length of the word library
    """
    words = read_words(path)
    return len(words)
