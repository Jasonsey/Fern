# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
""""model file"""
import tensorflow as tf
from tensorflow.keras import Model

from fern.setting import LOGGER


class BaseModel(object):
    """
    model builder

    Parameters
    ----------
    output_shape : dict[str, int], list[int], tuple[int]
        output shape without batch size
    max_seq_len : int
        the max input sequence length
    library_len : int
        the world library length
    initializer : str
        global initializer
    """
    def __init__(self, output_shape, max_seq_len, library_len, initializer='he_normal'):
        self.output_shape = output_shape
        self.max_seq_len = max_seq_len
        self.library_len = library_len
        self.initializer = initializer
        self.name = self.__class__.__name__

        self.model = self.build()
        self.print_summary()

    def print_summary(self):
        """
        print summary of model
        """
        summary = []
        self.model.summary(print_fn=summary.append)
        summary = '\n'.join(summary)
        LOGGER.warn(f"\n{summary}")

    def build(self):
        """
        build model

        Returns
        -------
        Model
            built model
        """
        raise NotImplementedError

    def save(self, path):
        """
        save model

        Parameters
        ----------
        path : str, pathlib.Path
            The model file path
        """
        self.model.save(path)

    def load(self, path):
        """
        load model

        Parameters
        ----------
        path : str, pathlib.Path
            The model file path
        """
        self.model = tf.keras.models.load_model(path)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    def trainable_variables(self):
        return self.model.trainable_variables
