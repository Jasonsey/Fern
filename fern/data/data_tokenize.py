# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""data tokenize"""
from typing import *
import re
from collections import Counter

import jieba
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer


SPECIAL_TOKEN = [
    '[pad]',    # 占位符
    '[cls]',    # 分类符
    '[sep]',    # 分割符号
    '[ukn]',    # 未知token
]


def str2word(string: str, zh_segmentation=True) -> List[str]:
    """
    分割字符串为词列表，支持的语言：中文、英文、数字、特殊符号（'-_.[]'），未支持的语言会被自动过滤

    操作顺序：
        1. 拆分中文和其他语言
        2. 除了中文、英文、数字、特殊符号（'-_.[]'）外，都作为分词分界线
        3. 针对中文：使用jieba分词
        4. 针对英文、数字：直接合成

    Args:
        string: the cleaned data
        zh_segmentation: 中文是否分词，默认使用结巴分词；否则按字分词

    Returns:
        cleaned word list
    """
    if zh_segmentation:
        string = re.sub(r'([\u4e00-\u9fa5]+)', r' \1 ', string)
        string = re.sub(r'(\[)(.+?)(])', r' \1\2\3 ', string)    # 确保特殊token的安全: xxx[yyy]zzz -> xxx [yyy] zzz

        string_list = re.split(r'[^a-zA-Z0-9\-_.\u4e00-\u9fa5\[\]]+', string)
        words = []
        for item in string_list:
            if not item:
                continue
            if re.match(r'[\u4e00-\u9fa5]', item):
                tmp = [word for word in jieba.cut(item) if word]
            else:
                tmp = [item]
            words.extend(tmp)
    else:
        string = re.sub(r'([\u4e00-\u9fa5])', r' \1 ', string)
        words = re.split(r'[^a-zA-Z0-9\-_.\u4e00-\u9fa5\[\]]+', string)
    return words


def generate_label_data(
        data: pd.DataFrame,
        label_col: str,
) -> Union[LabelBinarizer, MultiLabelBinarizer, Dict[str, Union[LabelBinarizer, MultiLabelBinarizer]]]:
    """
    利用sklearn工具加载标签字典，支持标签列的格式如下：
        1. 多任务：{'task1': [label1, label2], 'task2': label1}
        2. 单任务多标签： [label1, label2]
        3. 单任务单标签： label1

    Args:
        data: 带有标签列的data frame
        label_col: 标签列名字

    Returns:
        返回数据格式如下：
            1. 多任务：{'task1': MultiLabelBinarizer, 'task2': LabelBinarizer}
            2. 单任务多标签： MultiLabelBinarizer
            3. 单任务单标签： LabelBinarizer

    Raises:
        TypeError: 如果数据类型不正确，那么就会报错
    """
    label_example = data.loc[0, label_col]

    if isinstance(label_example, dict):
        # 多任务
        labels = {}

        def add_label(item):
            for key_ in item:
                if key_ in labels:
                    labels[key_].append(item[key_])
                else:
                    labels[key_] = [item[key_]]

        data[label_col].map(add_label)
        encoder = {}
        for key in labels:
            if isinstance(label_example[key], (list, tuple)):
                # 多标签
                encoder_ = MultiLabelBinarizer()
                encoder_.fit(labels[key])
                encoder[key] = encoder_
            elif isinstance(label_example[key], str):
                # 单标签
                encoder_ = LabelBinarizer()
                encoder_.fit(labels[key])
                encoder[key] = encoder_
    elif isinstance(label_example, (list, tuple)):
        # 单任务多标签
        labels = data[label_col].to_list()
        encoder = MultiLabelBinarizer()
        encoder.fit(labels)
    elif isinstance(label_example, str):
        # 单任务单标签
        labels = data[label_col].to_list()
        encoder = LabelBinarizer()
        encoder.fit(labels)
    else:
        raise TypeError('标签类型无法被支持')
    return encoder


def generate_word_library(data: pd.DataFrame, data_col: str, top: Optional[int] = None) -> Tuple[dict, dict]:
    """
    从数据集中加载词库，要求data_col列中的数据格式都是字符串

    Args:
        data: the data frame where the source data is stored.
        data_col: 数据加载源的列名，要求这个列里面都是字符串
        top: 按照词频排序，只提取前n个词。默认提取全部

    Returns:
        word2id, id2word字典

    Raises:
        ValueError: if no data provide, ValueError will be raised
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

    data[data_col].map(_map_func)
    counter = Counter(words)
    res = counter.most_common(n=top)

    res = [(token, 0) for token in SPECIAL_TOKEN] + res

    word2id = {item[0]: idx for idx, item in enumerate(res)}
    id2word = {idx: item[0] for idx, item in enumerate(res)}

    return word2id, id2word


def limit_token_length(tokens: list, /, n: int, strategy: str = 'tail') -> list:
    """
    限制token的长度.

    如果token列表长度超过n, 限制策略如下：
        1. head: 截取前n个token
        2. head+tail: 截取前n/3和后2n/3个token，并进行组合
        3. tail: 截取后n个token; 默认值

    注意：
        不建议在训练或者测试的时候移除token数量过多的样本，而是推荐截取的方式. 这样可以尽量训练和测试的样本格式的一致

    Args:
        tokens: 需要截取的token列表
        n: 最大列表长度
        strategy: 截取策略

    Returns:
        截取之后的token列表
    """
    if len(tokens) <= n:
        return tokens

    if strategy == 'head+tail':
        head = n//3
        res = tokens[:head] + tokens[head-n:]
    elif strategy == 'head':
        res = tokens[:n]
    else:
        res = tokens[-n:]
    return res
