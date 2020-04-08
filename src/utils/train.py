# Address Recognition
#
# Author: Max, Lin
# Email: max_lin1@dell.com
#
# =============================================================================
"""model trainer"""
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import LOGGER


class ModelTrainer(object):
    """model trainer"""
    def __init__(self, model, dataset_path, lr=0.003, batch_size=64):
        """
        model trainer

        Parameters
        ----------
        model : Model
            the model to be trained
        dataset_path : str, Path
            Points to training data in npz format
        lr : float
            learning rate
        batch_size : int
            batch size
        """
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(lr)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_acc = tf.keras.metrics.Mean(name='train_acc')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_acc = tf.keras.metrics.Mean(name='test_acc')

        self.ds_train, self.ds_test = self.load_dataset(dataset_path, batch_size)
        self.dataset_path = dataset_path
        self.label_weight = self.gen_label_weight()

    def train(self, epochs=1, early_stop=None):
        """
        train the model

        Parameters
        ----------
        epochs : int
            epoch number
        early_stop : int, optional
            If None, no early stop will be used. If is number, after number times no update, will early stop training.
        """
        best_score = 0
        number = 0
        for epoch in range(epochs):
            # At the beginning of the next epoch, reset the evaluation metrics
            self.train_loss.reset_states()
            self.train_acc.reset_states()
            self.test_loss.reset_states()
            self.test_acc.reset_states()
            number += 1

            for data, label in self.ds_train:
                self.train_step(data, label)

            for test_data, test_label in self.ds_test:
                self.test_step(test_data, test_label)

            score = self.test_acc.result()
            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Early Stop: {}/{}'
            LOGGER.info(template.format(
                epoch + 1,
                self.train_loss.result(),
                self.train_acc.result(),
                self.test_loss.result(),
                score,
                number,
                early_stop))

            if score > best_score:
                best_score = score
                number = 0
            if early_stop is not None and isinstance(early_stop, int) and number >= early_stop:
                break
        return best_score

    def loss(self, y_predicted, y_desired, train=True):
        """
        loss function

        Parameters
        ----------
        y_predicted : tf.Tensor
            with shape (None, 6, 10)
        y_desired : tf.Tensor
            with shape (None, 6, 10)
        train : bool
            True for training, False for testing

        Returns
        -------
        tf.Tensor
            with shape (1,)
        """
        res = - y_desired * tf.math.log(y_predicted + tf.keras.backend.epsilon())
        if train is True:
            res *= self.label_weight

        res = tf.reduce_max(res, axis=-1)
        res = tf.reduce_mean(res)
        return res

    @staticmethod
    def acc(y_predicted, y_desired, index=None):
        """
        accuracy function

        Parameters
        ----------
        y_predicted : tf.Tensor
            model output
        y_desired : tf.Tensor
            desired model output

        index : int, list[int], optional
            which postcode acc to be output

        Returns
        -------
        tf.Tensor
            with shape (None, 1)
        """
        mask = np.zeros(6, dtype=np.int64)
        if index is not None:
            mask[index] = 1
        else:
            mask[:] = 1
        y_ = tf.argmax(y_predicted, axis=-1)
        y = tf.argmax(y_desired, axis=-1)
        res = tf.reduce_sum((y - y_) * mask, axis=-1)
        res = tf.equal(res, tf.constant(0, tf.int64))
        return res

    @tf.function
    def train_step(self, data, label):
        """
        train one step

        Parameters
        ----------
        data : tf.Tensor
            the input train data
        label : tf.Tensor
            the output train label
        """
        with tf.GradientTape() as tape:
            predictions = self.model(data)
            loss = self.loss(predictions, label)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss.update_state(loss)

        acc = self.acc(predictions, label)
        self.train_acc.update_state(acc)

    @tf.function
    def test_step(self, data, label):
        """
        test one step

        Parameters
        ----------
        data : tf.Tensor
            the input test data
        label : tf.Tensor
            the output test labels
        """
        predictions = self.model(data)
        loss = self.loss(predictions, label, train=False)
        acc = self.acc(predictions, label)

        self.test_loss.update_state(loss)
        self.test_acc.update_state(acc)

    @staticmethod
    def load_dataset(path, batch_size):
        """
        load dataset from path

        Parameters
        ----------
        path : str, Path, optional
            Points to training data in npz format
        batch_size : int
            batch size

        Returns
        -------
        tuple[tf.data.Dataset]
            train and test dataset
        """
        with np.load(path) as data:
            data_train = data['data_train']
            label_train = data['label_train']

            data_test = data['data_test']
            label_test = data['label_test']
        dataset_train = tf.data.Dataset.from_tensor_slices((data_train, label_train))\
            .shuffle(len(data_train))\
            .batch(batch_size)
        dataset_test = tf.data.Dataset.from_tensor_slices((data_test, label_test)).batch(batch_size)
        return dataset_train, dataset_test

    def gen_label_weight(self):
        """
        read label weight from path

        Returns
        -------
        np.ndarray
            label weight
        """

        with np.load(self.dataset_path) as data:
            label_train = data['label_train']
            label_test = data['label_test']
        label = np.concatenate((label_train, label_test))
        weight = np.sum(label, axis=0)
        mask = np.where(weight == 0, np.float32(0), np.float32(1))
        weight += tf.keras.backend.epsilon()

        weight_by_postcode = np.array([1, 1, 2, 4, 5, 10], dtype=np.float32).reshape((-1, 1))
        weight_by_postcode /= np.sum(weight_by_postcode)        # for label weight by postcode

        weight = np.log(np.max(weight, axis=-1, keepdims=True) / weight) + 1
        weight *= mask
        return weight
