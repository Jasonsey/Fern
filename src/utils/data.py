# Fern
#
# Author: Jasonsey
# Email: 2627866800@qq.com
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

from config import LOGGER


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
            LOGGER.info(f'{path} does not exit. Creat it.')
            path.mkdir(parents=True)


class DataDownloader(BaseDataTool):
    """data downloader"""
    def __init__(self, host, user, password):
        """
        data loader

        Parameters
        ----------
        host : str
            sql server host
        user : str
            user name
        password : str
            user password
        """
        super().__init__()
        self.host = host
        self.user = user
        self.password = password

    def read_msssql(self, sql):
        conn = pymssql.connect(host=self.host, user=self.user, password=self.password, charset=r'utf8')
        data = pd.read_sql(sql, conn)
        self.data = data.dropna().reset_index(drop=True)
        conn.close()


class DataCleaner(BaseDataTool):
    """data cleaner"""
    def __init__(self, stop_words=None, user_words=None):
        """
        data cleaner

        Parameters
        ----------
        user_words : str, Path, optional
            user words path
        stop_words : str, Path, optional
            stop words path
        """
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

    def clean(self, data, update_data=True, input_col='data', output_col='label'):
        """
        clean data main entry

        Parameters
        ----------
        data : pd.DataFrame
            source data frame
        update_data : bool
            update self.data
        input_col : str
            The column name of the input data
        output_col : str
            The column name of the output label

        Returns
        -------
        pd.DataFrame
            cleaned data
        """
        for col in ['SVC_PSTL_CD', 'CUST_Address1', 'CUST_Address2', 'CUST_Address_line3']:
            if col not in data.columns:
                LOGGER.error(f'{col} should be in source data.columns')
                raise
        data[input_col] = data.apply(self.clean_func, axis=1)
        LOGGER.debug('All raw data has been cleaned.')

        data[input_col] = data[input_col].map(self.str2word)
        LOGGER.debug('All cleaned data has been split into word list.')

        data = data.dropna().reset_index(drop=True)
        data[output_col] = data['SVC_PSTL_CD']

        data = data[['SVC_DSPCH_ID', input_col, output_col]]
        if update_data:
            self.data = data
        return data

    def clean_func(self, items):
        """
        convert log

        Parameters
        ----------
        items : pd.Series
            one log

        Returns
        -------
        str
            cleaned log
        """
        addrs = [items['CUST_Address1'], items['CUST_Address2'], items['CUST_Address_line3']]
        postcode = str(items['SVC_PSTL_CD'])
        res = []
        for addr in addrs:
            addr = addr.strip()
            if postcode in addr:
                addr = re.sub(f'[\s\-:,.]*{postcode}[\s\-:,.]*', ' ', addr)
            if addr == '':
                continue
            res.append(addr)
        res = ','.join(res)
        res = res.lower()
        res = res.translate(self.table)
        res = re.sub('\s+', ' ', res)
        res = re.sub('\s*,+\s*', ',', res)
        return res

    def str2word(self, addr):
        """
        split a log into word list

        Parameters
        ----------
        addr : str
            address data

        Returns
        -------
        list[str]
            cleaned word list
        """
        data = nltk.word_tokenize(addr)
        data = self.tokenizer.tokenize(data)
        words = []
        for da in data:
            da = re.sub('[^a-zA-Z0-9\-_. ]', '', da)
            if len(da) > 1 and da != ' ' and da not in self.stop_words:
                words.append(da)

        if len(words) == 0:
            words = np.nan
        return words


class DataTransformer(BaseDataTool):
    """转换数据为神经网络输入格式"""
    PREFIX = [
        '<PAD>',    # 占位符
        # '<ST>',     # 开始字符
        # '<ED>'      # 终止字符
    ]

    def __init__(self,
                 word_path,
                 min_len=5,
                 max_len=25,
                 max_freq=3,
                 data=None,
                 output_shape=None,
                 input_col='data',
                 output_col='label',
                 filter_data=True):
        """
        data transformer

        Parameters
        ----------
        data : pd.DataFrame, optional
            The data frame where the address data is stored. If data no word library built, data should be provided.
        word_path : str
            Word library path
        max_freq : int
            The biggest word frequency
        min_len : int
            Minimum sentence length
        max_len : int
            Maximum sentence length
        output_shape : list[int]
            output shape of label
        input_col : str
            The column name of the input data
        output_col : str
            The column name of the output label
        filter_data : bool
            Whether to delete the non-conforming data
        """
        super().__init__()
        self.input_col = input_col
        self.output_col = output_col
        self.output_shape = output_shape
        self.min_len = min_len
        self.max_len = max_len
        self.max_freq = max_freq
        self.word_path = word_path

        self.word2id = self.read_word2id(data)
        self.filter_data = filter_data

    def transform(self, data, update_data=True):
        """
        transform data

        Parameters
        ----------
        data : pd.DataFrame
            The data frame where the address data is stored

        update_data : bool
            Whether to update self.data or not

        Returns
        -------
        np.ndarray
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
                'data': data,
                'label': label
            }
        return data, label

    def transform_data(self, data):
        """
        transform func

        Parameters
        ----------
        data : list[str]
            A single sequence of address words

        Returns
        -------
        np.ndarray
            The transformed word id sequence
        """
        data = [self.word2id[word] for word in data if word in self.word2id]
        data = pad_sequences([data], maxlen=self.max_len, padding='post')
        return data

    def transform_label(self, label):
        """
        transform func

        Parameters
        ----------
        label : np.int64
            A six-digit long integer

        Returns
        -------
        np.ndarray
            The transformed one-hot label
        """
        res = np.zeros([1] + self.output_shape, np.float32)
        for i in range(len(str(label))):
            number = int(str(label)[i])
            res[:, i, number] = 1.0
        return res

    def read_word2id(self, data=None):
        """
        read word library or reload word library

        Parameters
        ----------
        data : pd.DataFrame, optional
            The data frame where the address data is stored. If no word library built, data should be provided.

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
            The data frame where the address data is stored.

        Returns
        -------
        list[str]
            word library
        """
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
            The data frame where the address data is stored.

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


class DataSplitter(BaseDataTool):
    """split data into train data and test data"""
    def __init__(self, test_rate):
        super().__init__()
        self.test_rate = test_rate

    def split(self, data):
        """
        split function to split data

        Parameters
        ----------
        data : dict[str, np.ndarray]
            'data' and 'label' should be keys of the data
        """
        data, label = data['data'], data['label']
        assert len(data) == len(label)

        indexes = np.random.permutation(data.shape[0])
        i = int(data.shape[0] * self.test_rate)
        indexes_train, indexes_test = indexes[:i], indexes[i:]

        data_train, data_test = data[indexes_train], data[indexes_test]
        label_train, label_test = label[indexes_train], label[indexes_test]

        self.data = {
            'data_train': data_train,
            'data_test': data_test,
            'label_train': label_train,
            'label_test': label_test
        }

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
