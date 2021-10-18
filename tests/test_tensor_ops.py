# Fern
# 
# @Author: Lin, Max
# @Email : max_lin1@dell.com
# @Time  : 2021/9/27 12:44 下午
#
# =============================================================================
"""test_tensor_ops.py"""
import tensorflow as tf
from fern.utils import tensor_ops


def test_map_():
    tensor = tf.constant([[1, 2], [3, 4]])
    ragged_tensor = tf.ragged.constant([[1, 2], [3]])
    res_tensor = tensor_ops.map_flat_values(lambda x: x + 1, tensor)
    res_ragged_tensor = tensor_ops.map_flat_values(lambda x: x + 1, ragged_tensor)
    print(res_tensor)
    print(res_ragged_tensor)
