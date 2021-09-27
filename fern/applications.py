# Fern
# 
# @Author: Lin, Max
# @Email : max_lin1@dell.com
# @Time  : 2021/9/26 10:40 上午
#
# =============================================================================
"""applications.py"""
import tensorflow_text      # for hub.load
import tensorflow as tf
import tensorflow_hub as hub

from fern.logging import Logging
from fern.utils.tensor_ops import map_flat_values


logger = Logging()


class BertEncoder(tf.keras.layers.Layer):
    """BERT语言预训练模型, 默认使用中文"""
    _preprocessor = None
    _encoder = None
    model_config = {
        'bert_en_uncased_L-12_H-768_A-12': {
            'preprocessor_uri': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'encoder_uri': 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
        },
        'bert_en_uncased_L-24_H-1024_A-16': {
            'preprocessor_uri': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'encoder_uri': 'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4'
        },
        'bert_en_wwm_uncased_L-24_H-1024_A-16': {
            'preprocessor_uri': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'encoder_uri': 'https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/4'
        },
        'bert_en_cased_L-12_H-768_A-12': {
            'preprocessor_uri': 'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
            'encoder_uri': 'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/4'
        },
        'bert_en_cased_L-24_H-1024_A-16': {
            'preprocessor_uri': 'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
            'encoder_uri': 'https://tfhub.dev/tensorflow/bert_en_cased_L-24_H-1024_A-16/4'
        },
        'bert_en_wwm_cased_L-24_H-1024_A-16': {
            'preprocessor_uri': 'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
            'encoder_uri': 'https://tfhub.dev/tensorflow/bert_en_wwm_cased_L-24_H-1024_A-16/4'
        },
        'bert_zh_L-12_H-768_A-12': {
            'preprocessor_uri': 'https://tfhub.dev/tensorflow/bert_zh_preprocess/3',
            'encoder_uri': 'https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4'
        },
        'bert_multi_cased_L-12_H-768_A-12': {
            'preprocessor_uri': 'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
            'encoder_uri': 'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4'
        }
    }

    def __init__(self, seq_len: int, model_name: str, **kwargs):
        """
        根据配置生成Bert模型层

        Args:
            seq_len: 需要配置的最大输入长度, 默认128, 范围是1-512
            model_name: 在`BertEncoder.model_config.keys()`中选取
            **kwargs: keras layer通用配置
        """
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.model_name = model_name
        self.set_model(model_name)

        self.tokenize = self._preprocessor.tokenize
        self.bert_input_layer = hub.KerasLayer(
            self._preprocessor.bert_pack_inputs,
            arguments=dict(seq_length=seq_len),
            name='bert-input-layer')  # Optional argument.

    def get_config(self):
        config = super().get_config()
        config.update({
            'seq_len': self.seq_len,
            'model_name': self.model_name})
        return config

    def set_model(self, name):
        """
        修改模型和模型对应的preprocessor方法

        Args:
            name: 根据BertEncoder.uri_config.keys()获取合适的配置, 把key输入就可以完成模型的初始化
        """
        if name not in self.model_config:
            raise KeyError(f'无法找到合适的key: {self.uri_config.keys()}')
        preprocessor_uri = self.model_config[name]['preprocessor_uri']
        encoder_uri = self.model_config[name]['encoder_uri']
        self._preprocessor = hub.load(preprocessor_uri)
        self._encoder = hub.load(encoder_uri)

    def call(self, inputs, **kwargs):
        """
        批量文本转化为token

        Args:
            inputs: 支持输入任意shape的array, 支持ragged tensor; 例如 `[['text 1', 'text 2'], ['text 3']]`
            **kwargs:

        Returns:
            输入tensor则返回tensor, 输入ragged tensor则返回ragged tensor
        """
        encode_output = map_flat_values(self.map_func, inputs)
        return encode_output

    def map_func(self, task_logs):
        """
        输入string list, 输出bert转化后的输出

        Args:
            task_logs: shape [all_task_size, ], 所有当前batch的所有task log列表

        Returns:
            [all_task_size, encoder_length]
        """
        txt_tokenized = self.tokenize(task_logs)
        bert_input = self.bert_input_layer([txt_tokenized])
        bert_output = self._encoder(bert_input)['sequence_output']  # [task_size, seq_len, 768]
        res = bert_output[:, 0, :]  # [task_size, 768]
        return res
