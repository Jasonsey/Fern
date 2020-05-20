# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""data tools"""
import re
import csv
import typing

import nltk
import pymssql
import numpy as np
import pandas as pd
from pathlib import Path
from nltk.tokenize import MWETokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from fern.config import LOGGER


class BaseDataTool(object):
    def __init__(self):
        self.data = None    # type: typing.Optional[pd.DataFrame]

    def save(self, path):
        """
        save data to path

        Parameters
        ----------
        path : str, Path
            path where data save
        """
        self.check_path(path)
        if self.data is None:
            LOGGER.error('You should get source data before save')
            raise
        self.data.to_csv(path, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)

    def load(self, path):
        """
        load data from path

        Parameters
        ----------
        path : str, Path
            path where data save
        """
        data = pd.read_csv(path)
        self.data = data.dropna().reset_index(drop=True)

    @staticmethod
    def read_words(words_path):
        """
        read user words, stop words and word library from path

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
            with open(path, mode='r', encoding='utf-8') as f:
                res = f.readlines()
            res = [item.strip().lower() for item in res]
            return set(res)
        if words_path is None:
            words = set()
        else:
            words = read(words_path)
        words = list(words)
        return words

    @staticmethod
    def check_path(path):
        """
        check if path exits. If not exit, the path.parent will be created.

        Parameters
        ----------
        path : str, Path
            path to be check
        """
        path = Path(path).parent
        if not path.exists():
            LOGGER.warn(f'{path} does not exit. Creat it.')
            path.mkdir(parents=True)


class BaseDownloader(BaseDataTool):
    """
    data downloader

    Parameters
    ----------
    host : str
            sql server host
    user : str
        user name
    password : str
        user password
    """
    def __init__(self, host, user, password):
        super().__init__()
        self.host = host
        self.user = user
        self.password = password

    def read_msssql(self, sql):
        conn = pymssql.connect(host=self.host, user=self.user, password=self.password, charset=r'utf8')
        data = pd.read_sql(sql, conn)
        self.data = data.dropna().reset_index(drop=True)
        conn.close()


class BaseCleaner(BaseDataTool):
    """
    data cleaner

    Parameters
    ----------
    user_words : str, Path, optional
        user words path
    stop_words : str, Path, optional
        stop words path
    """
    def __init__(self, stop_words=None, user_words=None):
        super().__init__()
        self.stop_words = self.read_words(stop_words)

        nltk.download('punkt')
        user_words = self.read_words(user_words)
        user_words = [tuple(item.split(' ')) for item in user_words]
        self.tokenizer = MWETokenizer(user_words, separator=' ')

        self.table = {ord(f): ord(t) for f, t in zip(
            u'＊・，。！？【】（）％＃＠＆１２３４５６７８９０、；—：■　ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ',
            u'*·,.!?[]()%#@&1234567890.;-:￭ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        }

    def clean(self, data, update_data=True, input_col='data', output_col='label', idx_col=None):
        """
        clean data main entry

        Parameters
        ----------
        data : pd.DataFrame
            source data frame
        update_data : bool
            To save the cleaned data to self.data. The default is True.
        input_col : str
            The column name of the input data
        output_col : str
            The column name of the output label
        idx_col : str, optional
            The index column name for source data. If not provided, the output columns will not contain index column.

        Returns
        -------
        pd.DataFrame
            cleaned data
        """
        data[input_col] = data.apply(self.clean_data, axis=1)
        LOGGER.debug('All raw data has been cleaned.')
        data[input_col] = data[input_col].map(self.str2word)
        LOGGER.debug('All cleaned data has been split into word list.')

        data[output_col] = data.apply(self.clean_label, axis=1)
        LOGGER.debug('All label data has been cleaned.')

        columns = [idx_col, input_col, output_col] if idx_col is not None else [input_col, output_col]
        data = data[columns]
        data = data.dropna().reset_index(drop=True)
        if update_data:
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
        str, int
            cleaned label
        """
        raise NotImplementedError

    def clean_data(self, row):
        """
        convert input data function for pandas apply function

        Parameters
        ----------
        row : pd.Series
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
        data = nltk.word_tokenize(string)
        data = self.tokenizer.tokenize(data)
        words = []
        for da in data:
            da = re.sub('[^a-zA-Z0-9\-_. ]', '', da)
            if len(da) > 1 and da != ' ' and da not in self.stop_words:
                words.append(da)

        if len(words) == 0:
            words = np.nan
        return words


class BaseTransformer(BaseDataTool):
    """
    Data transformer to convert data to neural network input format

    Parameters
    ----------
    word_path : str
        Path to word library
    output_shape : list[int]
        output shape of label for transforming label
    min_len : int
        Minimum permissible sentence length for filtering training data
    max_len : int
        Maximum permissible sentence length for filtering training data and sequence padding
    min_freq : int
        Minimum permissible word frequency for making word library
    data : pd.DataFrame, optional
        The cleaned data frame only to make word library
        If the word library doesn't exit, you may not provide the data
    input_col : str
        The column name of the input data
    output_col : str
        The column name of the output label
    filter_data : bool
        Whether to delete the input data which the sentence length is greater than max_len or less than min_len
    """
    PREFIX = [
        '<PAD>',    # 占位符
        '<ST>',     # 开始字符
        '<ED>'      # 终止字符
    ]

    def __init__(self,
                 word_path,
                 output_shape,
                 min_len=5,
                 max_len=25,
                 min_freq=3,
                 data=None,
                 input_col='data',
                 output_col='label',
                 filter_data=True):
        super().__init__()
        self.input_col = input_col
        self.output_col = output_col
        self.output_shape = output_shape
        self.min_len = min_len
        self.max_len = max_len
        self.max_freq = min_freq
        self.word_path = word_path

        self.word2id = self.read_word2id(data)
        self.filter_data = filter_data

    def transform(self, data, update_data=True):
        """
        transform data

        Parameters
        ----------
        data : pd.DataFrame
            The data frame where the cleaned data is stored

        update_data : bool
            To save the cleaned data to self.data. The default is True.

        Returns
        -------
        tuple[np.ndarray]
            The transformed word id sequence
        """
        if self.filter_data:
            data = self.filter_data_func(data)

        data[self.input_col] = data[self.input_col].map(self.transform_data)
        data[self.output_col] = data[self.output_col].map(self.transform_label)

        data = data.dropna().reset_index(drop=True)
        data, label = np.concatenate(data[self.input_col]), np.concatenate(data[self.output_col])

        if update_data:
            self.data = {
                self.input_col: data,
                self.output_col: label
            }
        return data, label

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
        data = pad_sequences([data], maxlen=self.max_len, padding='post')       # pad after each sequence
        return data

    def transform_label(self, label):
        """
        label transforming function for pandas map function

        Parameters
        ----------
        label : np.int64
            to be transform source label data

        Returns
        -------
        np.ndarray
            The transformed label with shape (1, m, n, ..)
        """
        raise NotImplementedError

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
        if Path(self.word_path).exists():
            with open(self.word_path, 'r', encoding='utf-8') as f:
                words = f.readlines()
                words = [word.strip().lower() for word in words]
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
        _ = data[self.input_col].map(words.extend)
        words = pd.Series(words).value_counts()
        words = words[words >= self.max_freq]
        words = list(words.index)
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
        return data[data[self.input_col].map(__filter_func)].reset_index(drop=True)

    def save(self, path):
        """
        save data to path

        Parameters
        ----------
        path : str, Path
            path where data save
        """
        self.check_path(path)
        if self.data is None:
            LOGGER.error('You should get source data before save')
            raise
        np.savez(path, **self.data)

    def load(self, path):
        """
        load data from path

        Parameters
        ----------
        path : str, Path
            path where data save
        """
        self.data = dict(np.load(path))


class BaseSplitter(BaseDataTool):
    """split data into train data and val data"""
    def __init__(self, rate_val, random_state=None):
        super().__init__()
        self.rate_val = rate_val
        self.random_state = random_state

    def split(self, data, input_col='data', output_col='label'):
        """
        split function to split data

        Parameters
        ----------
        data : dict[str, np.ndarray]
            'data' and 'label' should be keys of the data
        input_col : str
            The data key name of the data dictionary
        output_col : str
            The label key name of the data dictionary

        Raises
        ------
        AssertionError
            If the data.shape[0] != label.shape[0]
        """
        data, label = data[input_col], data[output_col]
        assert len(data) == len(label)

        indexes = np.random.RandomState(self.random_state).permutation(data.shape[0])
        i = int(data.shape[0] * self.rate_val)
        indexes_val, indexes_train = indexes[:i], indexes[i:]

        data_train, data_val = data[indexes_train], data[indexes_val]
        label_train, label_val = label[indexes_train], label[indexes_val]

        self.data = {
            f'{input_col}_train': data_train,
            f'{input_col}_val': data_val,
            f'{output_col}_train': label_train,
            f'{output_col}_val': label_val
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
        self.check_path(path)
        if self.data is None:
            LOGGER.error('You should get source data before save')
            raise ValueError('You should get source data before save')
        np.savez(path, **self.data)

    def load(self, path):
        """
        load data from path

        Parameters
        ----------
        path : str, Path
            path where data save
        """
        self.data = dict(np.load(path))
