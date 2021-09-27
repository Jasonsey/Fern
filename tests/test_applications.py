# Fern
# 
# @Author: Lin, Max
# @Email : max_lin1@dell.com
# @Time  : 2021/9/27 12:43 下午
#
# =============================================================================
"""test_bert_encoder.py"""
import tensorflow_text
import tensorflow as tf
from fern.applications import BertEncoder


class TestBertEncoder(object):
    @classmethod
    def setup_class(cls):
        cls.model = BertEncoder(seq_len=4, model_name='bert_en_uncased_L-12_H-768_A-12')

    def test_keras_input(self):
        inp1 = tf.keras.layers.Input(shape=[], dtype=tf.string)
        inp2 = tf.keras.layers.Input(shape=[None, ], dtype=tf.string)
        inp3 = tf.keras.layers.Input(shape=[None, ], dtype=tf.string, ragged=True)
        oup1 = self.model(inp1)
        oup2 = self.model(inp2)
        oup3 = self.model(inp3)
        assert oup1.shape.as_list() == [None, 768]
        assert oup2.shape.as_list() == [None, None, 768]
        assert oup3.shape.as_list() == [None, None, 768]

    def test_tensor_input(self):
        inp1 = tf.constant(['a b c', 'd e f'])
        inp2 = tf.constant([['a b c', 'd e f']])
        inp3 = tf.ragged.constant([['a b c', 'd e f'], []])
        oup1 = self.model(inp1)
        oup2 = self.model(inp2)
        oup3 = self.model(inp3)
        assert oup1.shape.as_list() == [2, 768]
        assert oup2.shape.as_list() == [1, 2, 768]
        assert oup3.shape.as_list() == [2, None, 768]

    def test_model_summary(self):
        inp = tf.keras.layers.Input(shape=[], dtype=tf.string)
        oup = self.model(inp)
        model = tf.keras.Model(inputs=inp, outputs=oup)
        print(model.summary())
