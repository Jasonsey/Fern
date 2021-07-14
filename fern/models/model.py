# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
""""model file"""
import logging
from tensorflow.keras import Model, layers


logger = logging.getLogger()


class FernModel(object):
    model: Model

    def __init__(self, output_shape, max_seq_len, library_len, initializer='he_normal'):
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
        self.output_shape = output_shape
        self.max_seq_len = max_seq_len
        self.library_len = library_len
        self.initializer = initializer
        self.name = self.__class__.__name__

        self.model = self.build()
        self.print_summary()

        self.compile = self.model.compile
        self.fit = self.model.fit

    def print_summary(self):
        """
        print summary of model
        """
        summary = []
        self.model.summary(print_fn=summary.append)
        summary = '\n'.join(summary)
        logger.info(f"\n{summary}")

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
        self.model.load_weights(path)

    @property
    def predict(self):
        return self.model.predict

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    def trainable_variables(self):
        return self.model.trainable_variables


class TextCNN(FernModel):
    """
    References
    ----------
    Optimization of model Convolutional Neural Networks for Sentence Classification (https://arxiv.org/pdf/1408.5882.pdf)
    """
    def build(self):
        inp = layers.Input(shape=(self.max_seq_len,))
        x = layers.Embedding(self.library_len, 256, embeddings_initializer=self.initializer)(inp)
        x = layers.Conv1D(256, 5, padding='same', kernel_initializer=self.initializer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.GlobalMaxPool1D()(x)

        x = layers.Dense(128, kernel_initializer=self.initializer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        ys = []
        for key in self.output_shape:
            y = layers.Dense(self.output_shape[key],
                             kernel_initializer=self.initializer,
                             activation='softmax',
                             name=key)(x)
            ys.append(y)
        model = Model(inputs=inp, outputs=ys, name=self.name)    # if len(ys) == 1, than ys = ys[0]
        return model
