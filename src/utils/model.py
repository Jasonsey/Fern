# Fern
#
# Author: Jasonsey
# Email: 2627866800@qq.com
#
# =============================================================================
""""model file"""
from tensorflow.keras import Model

from config import LOGGER


class ModelBase(object):
    def __init__(self, output_shape, max_seq_len, library_len):
        """
        model builder

        Parameters
        ----------
        output_shape : list[int]
            output shape
        max_seq_len : int
            the max input sequence length
        library_len : int
            the world library length
        """
        self.output_shape = output_shape
        self.max_seq_len = max_seq_len
        self.library_len = library_len
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
        LOGGER.info(f"\n{summary}")

    def build(self):
        """
        build model and return

        Returns
        -------
        Model
            the model built
        """
        raise NotImplementedError
