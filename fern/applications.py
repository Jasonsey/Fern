# Fern
# 
# @Author: Fang, Bruce
# @Email : fscap@qq.com
# @Time  : 2021/10/09 12:03 下午
#
# =============================================================================
"""applications.py"""
import tensorflow_text
import tensorflow as tf
import tensorflow_hub as hub


class BertEncoder(tf.keras.layers.Layer):
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

    def __init__(self, seq_length=128, model_name: str = 'bert_en_uncased_L-12_H-768_A-12', trainable: bool = True,
                 **kwargs):
        super(BertEncoder, self).__init__(**kwargs)

        self.model_name = model_name

        if model_name not in self.model_config:
            raise KeyError(f'无法找到合适的key: {self.model_config.keys()}')
        preprocessor_uri = self.model_config[model_name]['preprocessor_uri']
        encoder_uri = self.model_config[model_name]['encoder_uri']

        preprocessor = hub.load(preprocessor_uri)
        #         self._preprocessor = hub.KerasLayer(preprocessor_uri)
        self.tokenize = hub.KerasLayer(preprocessor.tokenize)
        self.bert_pack_inputs = hub.KerasLayer(preprocessor.bert_pack_inputs, arguments=dict(seq_length=seq_length))

        self._encoder = hub.KerasLayer(encoder_uri, trainable=trainable, name='bert-encoder-layer')

    def call(self, inputs):
        tokenized_inputs = [self.tokenize(segment) for segment in [inputs]]
        tokens = self.bert_pack_inputs(tokenized_inputs)
        out = self._encoder(tokens)
        return out

    def get_config(self):
        config = super(BertEncoder, self).get_config()
        config.update({'model_name': self.model_name})
        return config

if __name__ == "__main__":
    inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name='input')
    layer = BertEncoder()
    outputs = layer(inputs)
    bert_encoder = tf.keras.Model(inputs=inputs, outputs=outputs, name='core_model')
    bert_encoder.save('my_h5_model')