# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""data tools"""
import re
import csv
import copy
import json
import pickle
import pathlib
from typing import *

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from tensorflow.keras.preprocessing.sequence import pad_sequences

from fern import setting
from fern.utils import common


class FernSeries(pd.Series):
    @property
    def _constructor(self):
        return FernSeries

    @property
    def _constructor_expanddim(self):
        return FernDataFrame


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

    def save(self, path):
        """
        save data frame with csv format to path

        Parameters
        ----------
        path : str, Path
            path where data save
        """
        common.check_path(path)
        self.to_csv(path, index=True, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)


class FernDownloader(object):
    def __init__(self, url):
        """
        data downloader

        Parameters
        ----------
        url: str
            The string form of the URL is dialect[+driver]://user:password@host/dbname[?key=value..]

        Examples
        --------
        downloader = FernDownloader('mysql+pymysql://user:passwd@hostname/dbname?charset=utf8mb4')
        """
        self.engine = create_engine(url)
        self.data: Optional[FernDataFrame] = None

    def read_sql(self, sql, index=None, drop=False):
        df = pd.read_sql(sql, self.engine)
        df = FernDataFrame(df)

        if drop:
            df = df.dropna()
        if index is None:
            self.data = df
        else:
            self.data = df.set_index(index)

    def save(self, path):
        """save data frame into csv"""
        self.data.save(path=path)


class FernCleaner(object):
    """
    data cleaner
    """

    def __init__(self,
                 stop_words=None,
                 cut_func=None,
                 update_data=True,
                 data_col='data',
                 label_col='label',
                 idx_col=None):
        """
        Parameters
        ----------
        stop_words : str, Path, optional
            stop words path
        cut_func : typing.Callable
            a function to split sequence to word list. If you want to use user_words, please define here
        update_data : bool
            To save the cleaned data to self.data. The default is True.
        data_col : str
            The column name of the input data
        label_col : str
            The column name of the output label
        idx_col : str, optional
            The index column name for source data. If not provided, the output columns will not contain index column.
        """
        self.data: Optional[FernDataFrame] = None
        self.logger = setting.LOGGER
        self.stop_words = common.read_regex_words(stop_words)
        self.update_data = update_data
        self.data_col = data_col
        self.label_col = label_col
        self.idx_col = idx_col

        if cut_func is None:
            self.cut_func = common.Sequence2Words(language='en', user_words=None)
        else:
            self.cut_func = cut_func
        self.table = {ord(f): ord(t) for f, t in zip(
            u'＊・，。！？【】（）％＃＠＆１２３４５６７８９０、；—：■　ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ',
            u'*·,.!?[]()%#@&1234567890.;-:￭ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')}

    def clean(self, data):
        """
        clean data main entry

        Parameters
        ----------
        data : pd.DataFrame
            source data frame

        Returns
        -------
        pd.DataFrame
            cleaned data with columns: data_col and label_col
        """
        if self.idx_col is not None and data.index.name != self.idx_col:
            data = data.set_index(self.idx_col)

        data[self.data_col] = data.apply(self.clean_data, axis=1)
        self.logger.debug('All raw data has been cleaned.')
        data[self.data_col] = data[self.data_col].map(self.str2word)
        self.logger.debug('All cleaned data has been split into word list.')

        data[self.label_col] = data.apply(self.clean_label, axis=1)
        self.logger.debug('All label data has been cleaned.')

        columns = [self.data_col, self.label_col]
        data = data[columns]
        data = data.dropna()
        data = FernDataFrame(data)
        if self.update_data:
            self.data = data
        return data

    def clean_label(self, row):
        """
        clean label function for pandas apply function

        Parameters
        ----------
        row : pd.Series
            find source label data

        Returns
        -------
        dict[str, list[str|int]]
            cleaned label
        """
        raise NotImplementedError

    def clean_data(self, row):
        """
        convert input data function for pandas apply function

        Parameters
        ----------
        row : pd.Series, dict
            the input data row

        Returns
        -------
        str
            cleaned log
        """
        raise NotImplementedError

    def str2word(self, string):
        """
        split a string data into word list

        Parameters
        ----------
        string : str
            the cleaned data

        Returns
        -------
        list[str]
            cleaned word list
        """
        data = self.cut_func(string)
        words = []
        for da in data:
            da = re.sub('[^a-zA-Z0-9\-_.\u4e00-\u9fa5<> ]', '', da)  # delete all unimportant words
            if len(da) > 1 and da != ' ' and not self.is_stop_words(da):
                words.append(da)

        if len(words) == 0:
            words = np.nan
        return words

    def is_stop_words(self, word):
        """
        check whether a word is a stop word by RegEx

        Parameters
        ----------
        word : str
            a word to be checked

        Returns
        -------
        bool
            whether a word is a stop word
        """
        for reg in self.stop_words:
            if reg.match(word):
                return True
        return False

    def save(self, path):
        """
        save data to path

        Parameters
        ----------
        path : str, Path
            path where data save
        """
        common.check_path(path)
        if self.data is None:
            self.logger.error('You should get source data before save')
            raise ValueError('You should get source data before save')
        self.data.save(path)

    def load(self, path):
        """
        load data from path

        Parameters
        ----------
        path : str, Path
            path where data save
        """
        data = pd.read_csv(path, index_col=self.idx_col)
        data[self.label_col] = data[self.label_col].map(eval)
        data[self.data_col] = data[self.data_col].map(eval)
        self.data = data


class FernTransformer(object):
    """
    Data transformer to convert data to neural network input format

    Parameters
    ----------
    word_path : str
        Path to word library
    label_path : str
        Path to label data
    output_shape : Optional[Dict[str, Union[List[int], int]]]
        output shape of label for transforming label.
        you don't have to defined it if you don't use it to transform label
    min_len : int
        Minimum permissible sentence length for filtering training data
    max_len : int
        Maximum permissible sentence length for filtering training data and sequence padding
    min_freq : int
        Minimum permissible word frequency for making word library
    data : pd.DataFrame, optional
        The cleaned data frame only to make word library
        If the word library doesn't exit, you may not provide the data
    data_col : str
        The column name of the input data
    label_col : str
        The column name of the output label
    filter_data : bool
        Whether to delete the input data which the sentence length is greater than max_len or less than min_len
    """
    PREFIX = [
        '<PAD>',  # 占位符
        '<ST>',  # 开始字符
        '<ED>',  # 终止字符
        '<SEP>'  # 分割符号
    ]

    def __init__(self,
                 word_path,
                 label_path,
                 output_shape=None,
                 min_len=5,
                 max_len=25,
                 min_freq=3,
                 data=None,
                 data_col='data',
                 label_col='label',
                 filter_data=True):
        self.data: Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]] = None

        self.data_col = data_col
        self.label_col = label_col
        self.output_shape = output_shape
        self.min_len = min_len
        self.max_len = max_len
        self.max_freq = min_freq
        self.word_path = word_path
        self.label_path = label_path

        self.word2id = self.read_word2id(data)
        self.filter_data = filter_data

        self.label_data = self.read_label_data(data)
        self.logger = setting.LOGGER

    def transform(self, data, update_data=True):
        """
        transform data frame

        Parameters
        ----------
        data : pd.DataFrame
            The data frame where the cleaned data is stored
        update_data : bool
            To save the cleaned data to self.data. The default is True.

        Returns
        -------
        tuple[np.ndarray, dict[str, np.ndarray]]
            The transformed word id sequence
        """
        if self.filter_data:
            data = self.filter_data_func(data)

        data[self.data_col] = data[self.data_col].map(self.transform_data)
        data[self.label_col] = data[self.label_col].map(self.transform_label)
        data = data.dropna()

        data_ = np.concatenate(data[self.data_col].to_list())

        labels = {}
        label = pd.DataFrame(data[self.label_col].to_list())  # data.label_col 每一行都是字典
        for col in label.columns:
            labels[col] = np.concatenate(label[col])

        if update_data:
            self.data = {
                self.data_col: data_,
                self.label_col: labels
            }
        return data, labels

    def transform_data(self, data):
        """
        data transforming function for pandas map function

        Parameters
        ----------
        data : list[str]
            A single sequence of words

        Returns
        -------
        np.ndarray
            The transformed word id sequence
        """
        data = [self.word2id[word] for word in data if word in self.word2id]
        data = pad_sequences([data], maxlen=self.max_len, padding='post')  # pad after each sequence
        return data

    def transform_label(self, label):
        """
        label transforming function for pandas map function

        Parameters
        ----------
        label : dict
            to be transform source label data

        Returns
        -------
        dict[str, np.ndarray]
            The transformed label.And Note that:
             - Please make sure using float32
             - Even the 2 category outputs An Array in a multi-category format
        """
        raise NotImplementedError

    def read_label_data(self, data):
        """
        read label data dict for label index

        Parameters
        ----------
        data : pd.DataFrame
            The data frame where the cleaned data is stored. If no label data built, data should be provided.

        Returns
        -------
        dict[str, list[str|int]]
            all label for index
        """
        if pathlib.Path(self.label_path).exists():
            with open(self.label_path, 'r') as f:
                label_data = json.load(f)
        else:
            label_data = self.reload_label_data(data)
        return label_data

    def reload_label_data(self, data):
        """
        reload label data

        The principle of building label data here is that labels are independent of each other

        Parameters
        ----------
        data : pd.DataFrame
            The data frame where the source data is stored.

        Returns
        -------
        dict[str, list[str|int]]
            label data dict

        Raises
        ------
        ValueError
            If no data provide, TypeError will be raised
        """
        if data is None:
            raise ValueError('No data is provided.')

        label_data = {}
        df_label = pd.DataFrame(data[self.label_col].to_list())  # data.label_col每一行都是字典
        for col in df_label.columns:
            tmp = set()
            df_label[col].map(lambda x: tmp.update(x))
            label_data[col] = list(tmp)

        with open(self.label_path, 'w') as f:
            json.dump(label_data, f)
        return label_data

    def read_word2id(self, data=None):
        """
        read word library or reload word library

        Parameters
        ----------
        data : pd.DataFrame, optional
            The data frame where the cleaned data is stored. If no word library built, data should be provided.

        Returns
        ----------
        dict[str, int]
            The word to id dictionary
        """
        if pathlib.Path(self.word_path).exists():
            words = common.read_words(self.word_path)
        else:
            words = self.reload_word_library(data)
        word2id = dict(zip(words, range(len(words))))
        return word2id

    def reload_word_library(self, data):
        """
        reload the word library

        Parameters
        -------
        data : pd.DataFrame
            The data frame where the source data is stored.

        Returns
        -------
        list[str]
            word library

        Raises
        ------
        ValueError
            If no data provide, TypeError will be raised
        """
        if data is None:
            raise ValueError('No data is provided.')

        words = []
        words_append = words.append

        def _map_func(word_list):
            for word in word_list:
                word = word.strip().lower()
                if word and word[0] != '#':
                    words_append(word)

        _ = data[self.data_col].map(_map_func)
        words = pd.Series(words).value_counts()
        words = words[words >= self.max_freq]
        words = list(words.index)

        words = [word for word in words if word.upper() not in self.PREFIX]

        words = self.PREFIX + words

        data = '\n'.join(words)
        with open(self.word_path, 'w') as f:
            f.write(data)
        return words

    def filter_data_func(self, data):
        """
        filter data to ensure the maximum and minimum length of the sequence

        Parameters
        -------
        data : pd.DataFrame
            The data frame where the cleaned data is stored.

        Returns
        -------
        pd.DataFrame
            the filtered data frame
        """

        def __filter_func(item):
            """
            to filter data

            Parameters
            ----------
            item : list[str]
                To be filtered data

            Returns
            -------
            bool
                Whether to detect the input item or not. If True, item should be kept, else should be deleted.
            """
            item = [word for word in item if word in self.word2id]
            return self.min_len <= len(item) <= self.max_len

        return data[data[self.data_col].map(__filter_func)]

    def save(self, path):
        """
        save data to path

        Parameters
        ----------
        path : str, Path
            path where data save
        """
        common.check_path(path)
        if self.data is None:
            self.logger.error('You should get source data before save')
            raise ValueError('You should get source data before save')
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)

    def load(self, path):
        """
        load data from path

        Parameters
        ----------
        path : str, Path
            path where data save
        """
        with open(path, 'rb') as f:
            self.data = pickle.load(f)


class FernSplitter(object):
    """split data into train data and val data"""

    def __init__(self, rate_val, random_state=None):
        self.data: Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]] = None

        self.rate_val = rate_val
        self.random_state = random_state
        self.logger = setting.LOGGER

    def split(self, data, data_col='data', label_col='label'):
        """
        split function to split data

        Parameters
        ----------
        data : dict[str, np.ndarray|dict[str, np.ndarray]]
            data_col and label_col should be keys of the data
        data_col : str
            The data key name of the data dictionary
        label_col : str
            The label key name of the data dictionary

        Raises
        ------
        AssertionError
            If the data.shape[0] != label.shape[0]
        """
        data_total, label_total = data[data_col], data[label_col]
        for _, label in label_total.items():
            assert len(data_total) == len(label)

        indexes = np.random.RandomState(self.random_state).permutation(data_total.shape[0])
        i = int(data_total.shape[0] * self.rate_val)
        indexes_val, indexes_train = indexes[:i], indexes[i:]

        data_train, data_val = data_total[indexes_train], data_total[indexes_val]
        label_train, label_val = {}, {}
        for label_name, label in label_total.items():
            label_train[label_name], label_val[label_name] = label[indexes_train], label[indexes_val]

        self.data = {
            # data
            f'{data_col}_total': data_total,
            f'{data_col}_train': data_train,
            f'{data_col}_val': data_val,
            # label
            f'{label_col}_total': label_total,
            f'{label_col}_train': label_train,
            f'{label_col}_val': label_val
        }

    def save(self, path):
        """
        save data to path

        Parameters
        ----------
        path : str, Path
            path where data save

        Raises
        ------
        ValueError
            You should get source data before save
        """
        common.check_path(path)
        if self.data is None:
            self.logger.error('You should get source data before save')
            raise ValueError('You should get source data before save')
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)

    def load(self, path):
        """
        load data from path

        Parameters
        ----------
        path : str, Path
            path where data save
        """
        with open(path, 'rb') as f:
            self.data = pickle.load(f)


class FernBalance(object):
    def __init__(self, rate=None, data_col='data', label_col='label', random_state=None):
        """
        Class for sample balance

        Parameters
        ----------
        rate : float, optional
            - Over-sample target number: max(number of label samples) * rate
            - If the number of label sample less than target number, then it will be over-sampled to target number
            - If rate = None, than nothing will be done
        data_col : str
            data column name
        label_col : str
            label column name
        random_state : int, optional
            random state
        """
        self.rate = rate
        self.data_col = data_col
        self.label_col = label_col
        self.random_state = random_state
        self.data: Optional[Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]] = None
        self.logger = setting.LOGGER

    def balance(self, data):
        """
        Balance data by label weight

        Parameters
        ----------
        data : dict[str, np.ndarray|dict[str, np.ndarray]]
            Dictionary data to be balanced, which contains keys:
             - data_total, data_train, data_val
             - label_total, label_train, label_val
        """
        if not self.rate:
            self.logger.warn('There is no rate provided for balance. Return source data directly.')
            return data
        data_total, label_total = self._balance(data[f'{self.data_col}_total'],
                                                list(data[f'{self.label_col}_total'].values()))
        label_total = dict(zip(data[f'{self.label_col}_total'].keys(), label_total))

        data_train, label_train = self._balance(data[f'{self.data_col}_train'],
                                                list(data[f'{self.label_col}_train'].values()))
        label_train = dict(zip(data[f'{self.label_col}_train'].keys(), label_train))

        self.data = {
            # data
            f'{self.data_col}_total': data_total,
            f'{self.data_col}_train': data_train,
            f'{self.data_col}_val': data[f'{self.data_col}_val'],
            # label
            f'{self.label_col}_total': label_total,
            f'{self.label_col}_train': label_train,
            f'{self.label_col}_val': data[f'{self.label_col}_val']
        }

    def _balance(self, data, labels):
        """
        Balance data and labels by weights of labels[0]

        Parameters
        ----------
        data : np.ndarray
            input array data
        labels : list[np.ndarray]
            output label array data

        Returns
        -------
        tuple[np.ndarray, list[np.ndarray]]
            Balanced data
        """
        res_data = copy.deepcopy(data)
        res_labels = copy.deepcopy(labels)
        label = labels[0]
        count_label = np.sum(label, axis=0)
        target_num = int(max(count_label) * self.rate)
        for col_idx, num in sorted(enumerate(count_label), key=lambda item: item[1]):
            row_idx = np.argwhere(label[:, col_idx] == 1).reshape(-1)
            row_idx = np.random.RandomState(self.random_state).permutation(row_idx)
            if num >= target_num or num < 2:
                continue
            repeat_times = int(target_num // num - 1)
            copy_num = int(target_num % num)

            _data: np.ndarray = data[row_idx]
            res_data = np.concatenate((res_data, _data.repeat(repeat_times, axis=0), _data[:copy_num]))

            for i in range(len(labels)):
                _label: np.ndarray = labels[i][row_idx]
                res_labels[i] = np.concatenate((res_labels[i], _label.repeat(repeat_times, axis=0), _label[:copy_num]))

        random_idx = np.random.RandomState(self.random_state).permutation(res_data.shape[0])
        res_data = res_data[random_idx]
        for i in range(len(res_labels)):
            res_labels[i] = res_labels[i][random_idx]
        return res_data, res_labels

    def save(self, path):
        """
        save data to path

        Parameters
        ----------
        path : str, Path
            path where data save

        Raises
        ------
        ValueError
            You should get source data before save
        """
        common.check_path(path)
        if self.data is None:
            self.logger.error('You should get source data before save')
            raise ValueError('You should get source data before save')
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)

    def load(self, path):
        """
        load data from path

        Parameters
        ----------
        path : str, Path
            path where data save
        """
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
