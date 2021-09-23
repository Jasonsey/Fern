# Fern
#
# Author: Jason, Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""loss"""
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.backend import epsilon
from tensorflow import math


class BinaryFocalLoss(Loss):
    """
    针对多标签分类/2分类的 focal loss
    """
    def __init__(self, gamma: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, clip_value_min=epsilon(), clip_value_max=1-epsilon())  # 限制范围[ep, 1-ep]
        y_ = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)  # 如果在这里使用
        loss = -((1 - y_) ** self.gamma) * math.log(y_)
        loss = math.reduce_sum(loss, axis=-1)
        return loss


class CategoricalFocalLoss(Loss):
    """
    针对多分类的 focal loss
    """
    def __init__(self, gamma: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, clip_value_min=epsilon(), clip_value_max=1-epsilon())  # 限制范围[ep, 1-ep]
        loss = -y_true * ((1 - y_pred)**self.gamma) * math.log(y_pred)
        loss = math.reduce_sum(loss, axis=-1)
        return loss
