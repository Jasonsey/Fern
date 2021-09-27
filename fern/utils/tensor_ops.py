# Fern
# 
# @Author: Lin, Max
# @Email : max_lin1@dell.com
# @Time  : 2021/9/27 12:19 下午
#
# =============================================================================
"""tensor_ops.py"""
from typing import Union, Callable
import tensorflow as tf


# @tf.function
def map_flat_values(map_fn: Callable, tensor: Union[tf.Tensor, tf.RaggedTensor]):
    """
    对于每个tensor, 实现flatten再传值, 然后保持前后shape一致

    Args:
        map_fn: map函数
        tensor: 输入的tensor

    Returns:
        处理好的tensor
    """
    if isinstance(tensor, tf.RaggedTensor):
        output = tf.ragged.map_flat_values(map_fn, tensor)
    else:
        # Tensor
        before_shape = tensor.shape.as_list()

        tensor = tf.reshape(tensor, (-1,))
        output = map_fn(tensor)

        after_shape = before_shape + output.shape.as_list()[1:]
        if after_shape[-1] == 1:
            # 如果最后一个维度是1, 那么会清理它
            after_shape.pop(-1)
        for i in range(len(after_shape)):
            # shape
            if after_shape[i] is None:
                after_shape[i] = -1
        output = tf.reshape(output, after_shape)
    return output
