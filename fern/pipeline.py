# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""存放部分模型现成的数据流水线功能"""
from typing import *
from pathlib import Path
from collections import defaultdict

import pandas as pd
import tensorflow as tf

from fern.data import read_words, train_data_split, str2word


class BaseDataPipeline(object):
    batch_size: Optional[int] = None

    def get_dataset(self, training: bool) -> tf.data.Dataset:
        raise NotImplementedError

    @property
    def train_ds(self):
        return self.get_dataset(training=True)

    @property
    def val_ds(self):
        return self.get_dataset(training=False)


class Word2VecDataPipeline(BaseDataPipeline):
    """训练word2vec向量(skip-gram)，做了如下工作：

    1. 通过输入的数据创建数据流水线
    2. 实现停用词过滤
    3. 实现分词
    4. 实现大小写配置
    5. 创建词库, 忽略词频<2的单词

    """
    word2id: Optional[Dict] = None
    unk_token = '[UNK]'

    def __init__(self,
                 data: List[str],
                 batch_size: int,
                 stop_words: str = None,
                 zh_segmentation: bool = True,
                 lower: bool = True,
                 win_size: int = 2,
                 random_seed: Optional[int] = None,
                 ):
        """
        Args:
            data: 原始数据列表
            batch_size: 批大小
            stop_words: 停用词路径
            zh_segmentation: 中文是否使用分词工具分词
            lower: 是否忽略大小写
            win_size: 生成(target, context)的窗口大小
            random_seed: 随机种子，用于复现
        """
        self._data = pd.DataFrame({'source': data})
        if lower:
            self._data['source'] = self._data['source'].map(lambda x: x.lower())

        if stop_words is None:
            stop_words = Path(__file__).parent / 'config/stop_words.txt'
        self.stop_words = read_words(stop_words)
        self.word2id, self.id2word = self._build_vocab(zh_segmentation)
        self.vocab_size = len(self.word2id)

        # 分词，词转id，分数据集
        self._data['tokens'] = self._data['source'].map(lambda x: self.tokenize(x, zh_segmentation))
        self._data['ids'] = self._data['source'].map(self.token2id)
        self.train_data, self.val_data = train_data_split(self._data, test_size=0.2, random_state=random_seed)

        # 数据生成其他配置
        self.win_size = win_size
        self.batch_size = batch_size

    def _build_vocab(self, zh_segmentation: bool):
        """制作的词库按照词频从大到小排列, UNK放在最后一个字符"""
        def map_func(sentence: str):
            words = str2word(sentence, zh_segmentation)
            for _word in words:
                if _word in self.stop_words:
                    continue
                vocab[_word] += 1

        vocab = defaultdict(int)
        self._data['source'].map(map_func)
        vocab = {word: value for word, value in vocab.items() if value > 1}
        vocab = sorted(vocab, key=lambda key: vocab[key], reverse=True)
        vocab = list(vocab) + [self.unk_token]
        word2id = {word: idx for idx, word in enumerate(vocab)}
        id2word = {idx: word for idx, word in enumerate(vocab)}
        return word2id, id2word

    def tokenize(self, sentence: str, zh_segmentation: bool) -> List[str]:
        """把句子转化为token数据，如果不在词库中，那么会替换为UNK"""
        assert self.word2id is not None
        words = str2word(sentence, zh_segmentation)
        for i in range(len(words)):
            if words[i] not in self.word2id:
                words[i] = self.unk_token
        return words

    def token2id(self, words: List[str]) -> List[int]:
        """把词列表转化为token列表"""
        ids = [self.word2id.get(word, self.word2id[self.unk_token]) for word in words]
        return ids

    def data_generator(self, data: pd.DataFrame):
        """生成(target_id, context_id)"""
        data = data.sample(frac=1)
        for index, row in data.iterrows():
            # 每行一个句子，根据win_size从句子中生成（target, context）
            ids = row['ids']
            for i in range(len(ids)):
                index_min = max(i - self.win_size, 0)
                index_max = min(i + self.win_size, len(ids) - 1)
                for ii in range(index_min, index_max+1):
                    if ii == i:
                        continue
                    yield ids[i], [ids[ii]]

    def get_dataset(self, training: bool):
        """
        把生成器转化为tensorflow数据集

        Args:
            training: True返回训练数据集，如果是False返回测试数据集

        Returns:
            tensorflow数据集
        """
        if training:
            data = self.train_data
        else:
            data = self.val_data
        target_id = tf.TensorSpec(shape=(), dtype=tf.int64)
        context_id = tf.TensorSpec(shape=(1,), dtype=tf.int64)
        output_signature = (target_id, context_id)
        dataset = tf.data.Dataset.from_generator(
            generator=lambda: self.data_generator(data),
            output_signature=output_signature)
        dataset = dataset.cache().shuffle(self.batch_size * 10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
