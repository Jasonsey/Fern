# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
""""model file"""
import logging
import tensorflow as tf
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
    Optimization of model Convolutional Neural Networks for Sentence Classification
     (https://arxiv.org/pdf/1408.5882.pdf)
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


class Word2Vec(tf.Module):
    """
    根据论文在TF2.0上复现word2vec模型, 这里主要复现Skip-Gram模型, 该模型得到的结论：

    1. 当前词和附近词相似（余弦相似度）
    2. 根据这个训练方法得到的词向量直接相互加减可以表达部分语义


    References:
        1. [TF2.0 word2vec](https://www.tensorflow.org/tutorials/text/word2vec#subclassed_word2vec_model)

    """
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128, num_ns: int = 4):
        super().__init__(name='word2vec')
        self.vocab_size = vocab_size
        self.num_ns = num_ns
        self.target_embedding = layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=1,
            name="w2v_embedding")
        self.context_weight = tf.Variable(tf.random.normal((vocab_size, embedding_dim)), name='context_weight')
        self.context_bias = tf.Variable(tf.random.normal((vocab_size, )), name='context_bias')  # embed的需要为每个词添加bias

    @tf.function
    def __call__(self, target):
        """
        计算词列表的词向量

        Args:
            target: 一维的词id列表, shape=(batch,)

        Returns:
            返回shape=(batch, embed)的词向量
        """
        target = self.target_embedding(target)  # (batch, embed)
        return target

    @tf.function
    def loss(self, target, context, training=None):
        """训练时使用NCE loss，否则使用sigmoid cross entropy loss

        Args:
            target: 经过embedding后的词向量
            context: 上下文词的编号，起始可以理解为label
            training: 如果训练，返回nce loss，否则返回sigmoid loss

        Returns:
            返回shape=()的loss

        """
        # target, context  # (batch, embed)  (batch,)
        if training:
            loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.context_weight, biases=self.context_bias, labels=context,
                                                 inputs=target, num_sampled=self.num_ns, num_classes=self.vocab_size))
        else:
            logits = tf.matmul(target, tf.transpose(self.context_weight))
            logits = tf.nn.bias_add(logits, self.context_bias)
            labels_one_hot = tf.one_hot(context, self.vocab_size)
            labels_one_hot = tf.squeeze(labels_one_hot, axis=1)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_one_hot,
                logits=logits)
            loss = tf.reduce_mean(loss)
        return loss

    def save(self, model_dir):
        """保存词向量到本地"""
        tf.saved_model.save(self, str(model_dir))

    @classmethod
    def load(cls, model_dir):
        """从配置文件中加载重新加载模型"""
        model = tf.saved_model.load(model_dir)
        return model
