# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""model trainer"""
import typing
import pickle
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric

from fern.setting import LOGGER
from .common import ProgressBar
from .model import FernModel


class FernTrainer(object):
    """
    model trainer

    This can be used in one output and multi outputs

    Parameters
    ----------
    model : FernModel
        the model to be trained
    path_data : str, Path
        Points to training data in npz format
    opt : str
        the name of optimizer
    lr : float
        learning rate
    batch_size : int
        batch size
    """
    def __init__(self, model, path_data, opt='adam', lr=0.003, batch_size=64):
        self.model = model
        self.optimizer = tf.keras.optimizers.get({
            'class_name': opt,
            'config': {'learning_rate': lr}
        })

        self.metrics, self.early_stop_metric = self.setup_metrics()

        self.data = self.load_data(path_data, batch_size)
        self.label_weight = self.get_label_weight(path_data)

    def train(self, epochs=1, early_stop=None, mode='search'):
        """
        train the model

        Parameters
        ----------
        epochs : int
            epoch number
        early_stop : int, optional
            - If None, no early stop will be used.
            - If is number, after number times no update, will stop training early.
        mode : str
            - search: train model with data_train
            - server: train model with data_train and data_val

        Returns
        -------
        (float, int)
            (best_score, best_epoch)
        """
        if mode == 'server':
            dataset = self.data['dataset_total']
            total = self.data['step_total']
        else:
            dataset = self.data['dataset_train']
            total = self.data['step_train']

        best_score = 0
        best_epoch = 0
        number = 0
        for epoch in range(epochs):
            # reset metrics before the next epoch
            for stage in self.metrics:
                for metric in self.metrics[stage]:
                    self.metrics[stage][metric].reset_states()
            number += 1

            for items_train in ProgressBar(dataset, desc='Train: ', total=total):
                self.train_step(*items_train)

            for items_val in ProgressBar(self.data['dataset_val'], desc='Val: ', total=self.data['step_val']):
                self.val_step(*items_val)

            stop_flag = False
            if early_stop is None:
                log = [f'Epoch: {epoch}']
            else:
                early_stop_result = self.early_stop_metric.result().numpy()
                if hasattr(early_stop_result, '__len__'):
                    score = early_stop_result[-1]
                else:
                    score = early_stop_result
                if score - best_score >= 0.0001:
                    best_score = score
                    best_epoch = epoch + 1
                    number = 0
                elif number == early_stop:
                    stop_flag = True
                log = [f"Epoch: {epoch}, Early Stop: {number}/{early_stop}({'Yes' if stop_flag else 'No'})"]

            for stage in self.metrics:
                log_line = [f'{stage}: ']
                for metric in self.metrics[stage]:
                    result = self.metrics[stage][metric].result()
                    log_line.append(f'{metric}: {result}')
                log_line = '\t'.join(log_line)
                log.append(log_line)
            log = '\n\t'.join(log)
            log += '\n'
            LOGGER.warn(log)

            if stop_flag:
                break
        return best_score, best_epoch

    @tf.function
    def train_step(self, data, *labels):
        """
        train one step

        Parameters
        ----------
        data: tf.Tensor
            the input train data
        labels: tuple[tf.Tensor]
            the output train labels
        """
        label = labels[0] if len(labels) == 1 else list(labels)
        with tf.GradientTape() as tape:
            prediction = self.model(data)
            loss = self.loss(prediction, label)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        acc = self.acc(prediction, label)

        self.metrics['train']['loss'].update_state(loss)
        self.metrics['train']['acc'].update_state(acc)

    @tf.function
    def val_step(self, data, *labels):
        """
        val one step

        Parameters
        ----------
        data: tf.Tensor
            the input train data
        labels: tuple[tf.Tensor]
            the output train labels
        """
        label = labels[0] if len(labels) == 1 else list(labels)
        prediction = self.model(data)
        loss = self.loss(prediction, label, with_label_weight=False)
        acc = self.acc(prediction, label)

        self.metrics['val']['loss'].update_state(loss)
        self.metrics['val']['acc'].update_state(acc)

    @staticmethod
    def setup_metrics():
        """
        Initialize the evaluation metrics.
        If you define the metric yourself, please remember to edit your train_step and val_step

        Returns
        -------
        (dict[str, dict[str, Metric]], Metric)
            The dictionary with train and test, and each train or test with a dictionary of metric.
            And the other output of metric is for early stop.

        Examples
        --------
        ```python

        # Possible return format pseudo-code

        res = (
            {
                'train': {
                    'loss': MetricLoss,
                    'acc': MetricAcc,
                    'you_defined': ...},
                'val': {
                    'loss': MetricLoss,
                    'acc': MetricAcc,
                    'you_defined': ...},
            },
            EarlyStopMetric    # default is val_acc
        )

        ```
        """
        res = {
            'train': {
                'loss': tf.keras.metrics.Mean(name='train_loss'),
                'acc': tf.keras.metrics.Mean(name='train_acc')
            },
            'val': {
                'loss': tf.keras.metrics.Mean(name='val_loss'),
                'acc': tf.keras.metrics.Mean(name='val_acc'),
            }
        }
        return res, res['val']['acc']

    def loss(self, ys_predicted, ys_desired, with_label_weight=True):
        """
        loss function

        This function is for one output, multi outputs and for one-hot prediction.
        If you want to use multi-label prediction, you should rewrite this function.

        Parameters
        ----------
        ys_predicted : list[tf.Tensor], tf.Tensor
            each with shape (None, m)
        ys_desired : list[tf.Tensor], tf.Tensor
            each with shape (None, m)
        with_label_weight : bool
            True for multiplying loss by label weight

        Returns
        -------
        tf.Tensor
            with shape (1, )
        """
        if isinstance(ys_predicted, list):
            assert len(ys_predicted) == len(ys_desired)
        else:
            ys_predicted = [ys_predicted]
            ys_desired = [ys_desired]
        sum_res = 0
        for i in range(len(ys_predicted)):
            res = - ys_desired[i] * tf.math.log(ys_predicted[i] + tf.keras.backend.epsilon())
            if with_label_weight is True:
                res *= self.label_weight

            res = tf.reduce_max(res, axis=-1)
            res = tf.reduce_mean(res)
            sum_res += res
        return sum_res

    @staticmethod
    def acc(ys_predicted, ys_desired):
        """
        accuracy function

        This function is for one output, multi outputs and for one-hot prediction.
        If you want to use multi-label prediction, you should rewrite this function.

        Parameters
        ----------
        ys_predicted : list[tf.Tensor], tf.Tensor
            each with shape (None, m)
        ys_desired : list[tf.Tensor], tf.Tensor
            each with shape (None, m)

        Returns
        -------
        tf.Tensor
            with shape (None, 1). When multi outputs, only while all outputs is True, the response will be True
        """
        if isinstance(ys_predicted, list):
            assert len(ys_predicted) == len(ys_desired)
        else:
            ys_predicted = [ys_predicted]
            ys_desired = [ys_desired]
        res = 1
        for i in range(len(ys_predicted)):
            y_ = tf.argmax(ys_predicted[i], axis=-1)
            y = tf.argmax(ys_desired[i], axis=-1)
            res *= tf.cast(tf.equal(y, y_), tf.float32)
        return res

    @staticmethod
    def load_data(path, batch_size):
        """
        load dataset from path

        if there are multiple labels, such as data and label1, label2, output dataset with be (data, label1, label2)

        Parameters
        ----------
        path : str, Path, optional
            Points to training data in npz format
        batch_size : int
            batch size

        Returns
        -------
        dict[str, tf.data.Dataset|int]
            a dictionary with dataset_train, dataset_val, dataset_total, step_train, step_val and step_total
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            data_total: np.ndarray = data['data_total']
            label_total: typing.Dict[str, np.ndarray] = data['label_total']

            data_train: np.ndarray = data['data_train']
            label_train: typing.Dict[str, np.ndarray] = data['label_train']

            data_val: np.ndarray = data['data_val']
            label_val: typing.Dict[str, np.ndarray] = data['label_val']

        dataset_train = tf.data.Dataset.from_tensor_slices(tuple([data_train] + list(label_train.values()))) \
            .shuffle(len(data_train)) \
            .batch(batch_size)
        dataset_val = tf.data.Dataset.from_tensor_slices(tuple([data_val] + list(label_val.values()))).batch(batch_size)
        dataset_total = tf.data.Dataset.from_tensor_slices(tuple([data_total] + list(label_total.values())))\
            .shuffle(len(data_total)).batch(batch_size)

        step_train = int(np.ceil(len(data_train) / batch_size))
        step_val = int(np.ceil(len(data_val) / batch_size))
        step_total = int(np.ceil(len(data_total) / batch_size))

        data = {
            'dataset_train': dataset_train,
            'dataset_val': dataset_val,
            'dataset_total': dataset_total,
            'step_train': step_train,
            'step_val': step_val,
            'step_total': step_total,
        }
        return data

    @staticmethod
    def get_label_weight(path):
        """
        read label weight from the path where data in npz format was stored

        Parameters
        ----------
        path : str, Path, optional
            Points to training data in npz format

        Returns
        -------
        dict[str, np.ndarray], np.ndarray
            If input shape is (None, m), the label weight shape will be (m, ).
            - dict for multi outputs
            - array for one output
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            label_total: typing.Dict[str, np.ndarray] = data['label_total']

        weights = {}
        for key in label_total:
            weight = np.sum(label_total[key], axis=0)
            mask = np.where(weight == 0, 0., 1.)
            weight += tf.keras.backend.epsilon()

            weight = np.log(np.max(weight, axis=-1, keepdims=True) / weight) + 1
            weight *= mask
            weights[key] = weight
        if len(weights) == 1:
            weights = list(weights.values())[0]
        return weights

    def save(self, path):
        """
        save model

        Parameters
        ----------
        path : str, Path
            The model home path
        """
        path = str(pathlib.Path(path) / f'{self.model.name}.h5')
        self.model.save(path)

    def load(self, path):
        """
        load model

        Parameters
        ----------
        path : str, Path
            The model home path
        """
        path = str(pathlib.Path(path) / f'{self.model.name}.h5')
        self.model = tf.keras.models.load_model(path)
