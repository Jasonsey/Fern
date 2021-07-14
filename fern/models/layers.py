# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""layer model"""
import typing
import re
import pathlib

import tensorflow as tf
import tensorflow_text as text
from tensorflow.keras import layers


class Conv1DPassMask(layers.Conv1D):
    """to pass mask though conv1d"""
    def compute_mask(self, inputs, mask=None):
        return mask


class FlattenPassMask(layers.Flatten):
    """传递mask"""
    def compute_mask(self, inputs, mask=None):
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.concat([mask] * inputs.shape[-1], axis=-1)
        mask = tf.reshape(mask, (-1, mask.shape[-2] * mask.shape[-1]))
        return mask


class DenseWithMask(layers.Dense):
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.keras.backend.floatx())
            inputs *= mask
            inputs /= tf.reduce_sum(mask, axis=-1, keepdims=True)
        res = super().call(inputs)
        return res


class AttentionLayer(tf.keras.layers.Layer):
    """to replace global max pool"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weight = None  # type: typing.Optional[tf.keras.layers.Layer]

    def build(self, input_shape):
        """input_shape = (None, 25, 256)"""
        assert len(input_shape) == 3, str(input_shape)
        self.weight = self.add_weight(name='att_weight',
                                      shape=(input_shape[-2], input_shape[-2]),
                                      initializer='uniform',
                                      trainable=True)
        super().build(input_shape)

    def call(self, inputs, mask=None):
        """inputs.shape = (None, 25, 256)"""
        w = tf.transpose(inputs, perm=(0, 2, 1))
        w = tf.linalg.matmul(w, self.weight)
        w = tf.math.softmax(w)
        outputs = tf.linalg.matmul(w, inputs)
        outputs = tf.math.reduce_sum(outputs, axis=-2)
        return outputs

    def compute_output_shape(self, input_shape):
        res = input_shape[:-2] + input_shape[-1]
        return res

    def compute_mask(self, inputs, mask=None):
        return mask


class ScaledDotProductAttention(layers.Layer):
    def __init__(self, units=64, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        """input_shape = (None, 25, 256)"""
        super().build(input_shape)

    def call(self, inputs, mask=None):
        """inputs.shape = (None, 25, 256)"""
        q = k = v = inputs
        feature_dim = inputs.shape[-1]

        kt = tf.transpose(k, perm=(0, 2, 1))
        res = tf.linalg.matmul(q, kt)
        res = res / (feature_dim ** 0.5)
        if mask is not None:
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.cast(mask, dtype=tf.keras.backend.floatx())
            res *= mask
        res = tf.math.softmax(res)
        res = tf.linalg.matmul(res, v)
        return res

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask


class TransposeLayer(layers.Layer):
    """Transpose Layer"""
    def call(self, inputs, mask=None):
        y = tf.transpose(inputs, perm=(0, 2, 1))
        return y

    def compute_output_shape(self, input_shape):
        res = [input_shape[0], input_shape[2], input_shape[1]]
        return res


class BertTokenizer(tf.Module):
    """
    根据vocab字典转化token
    """
    def __init__(self, vocab_path, suffix_indicator='##', name='bert-tokenizer'):
        super().__init__(name=name)
        self.tokenizer = text.BertTokenizer(vocab_path, suffix_indicator=suffix_indicator, lower_case=True)
        self._suffix_indicator = suffix_indicator
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)

    def init_signature(self):
        # 看似没有效果, 这样就可以实现模型的自动初始化了(可以使用tf save_model导出)
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        enc = enc.merge_dims(-2, -1)
        return enc

    @tf.function
    def detokenize(self, tokenized):
        txt_tokens = tf.gather(self.vocab, tokenized)
        # Join with spaces.
        words = tf.strings.reduce_join(txt_tokens, separator=' ', axis=-1)
        words = tf.strings.regex_replace(words, re.escape(' ' + self._suffix_indicator))
        return words

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        mask = tf.strings.regex_full_match(self.vocab, r'\[[A-Z]+\]')
        res = tf.boolean_mask(self.vocab, mask)
        return res
