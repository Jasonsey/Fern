# Fern
#
# Author: Jason, Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""metric"""
import tensorflow as tf
from tensorflow.keras.metrics import Metric


class BinaryCategoricalAccuracy(Metric):
    """
    To calculate the accuracy of the m outputs, when the output is multi-output matrix with shape (None, m, n)

    Parameters
    ----------
    m : int
        the number of output
    """
    def __init__(self, m, name='binary_categorical_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.count = self.add_weight('count', initializer='zeros')
        self.values = self.add_weight('values', shape=(m, ), initializer='zeros')
        self.m = m

    def update_state(self, y_true, y_pred):
        """
        update state

        Parameters
        ----------
        y_true : tf.Tensor
            with shape (None, m, n)
        y_pred : tf.Tensor
            with shape (None, m, n)
        """
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

        value = tf.equal(y_true, y_pred)
        value = tf.cast(value, dtype=self.dtype)
        value = tf.reduce_sum(value, axis=0)

        self.values.assign_add(value)
        self.count.assign_add(y_true.shape[0])
    
    def reset_states(self):
        self.count.assign(0)
        self.values.assign([0] * self.m)

    def result(self):
        """
        return mean result
        """
        return self.values / self.count
