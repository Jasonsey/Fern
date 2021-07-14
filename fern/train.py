# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""model trainer"""
import logging
import pickle
from pathlib import Path
from typing import *

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric

from fern.models.model import FernModel
from fern.utils.common import ProgressBar


logger = logging.getLogger('Fern')


class FernBaseTrainer(object):
    model: FernModel
    optimizer: Optional[tf.optimizers.Optimizer]
    data: Dict[str, Union[tf.data.Dataset, int]]
    metrics: Dict[str, Dict[str, Metric]]
    early_stop_metric: Metric
    label_weight: Union[Dict[str, np.ndarray], np.ndarray]

    def train(self, epochs: int = 1, early_stop: Optional[int] = None, mode: str = 'search') -> Tuple[float, int]:
        """
        train the model

        Args:
            epochs: epoch number
            early_stop:
                If None, no early stop will be used.
                If is number, after number times no update, will stop training early.
            mode:
                search: train model with data_train
                server: train model with data_train and data_val

        Returns:
            (best_score, best_epoch)
        """
        raise NotImplementedError

    @staticmethod
    def load_data(
            path: Union[str, Path],
            batch_size: int,
            data_col: str,
            label_col: str,
    ) -> Dict[str, Union[tf.data.Dataset, int]]:
        """
        load dataset from path

        if there are multiple labels or data, for example data={'col1': data1, 'col2': data2}
        and label={'col3': label1, 'col4': label2}, then the output of the data set will be dict too, which looks like
        data_batch={'col1': data1, 'col2': data2} and label_batch={'col3': label1, 'col4': label2}

        Args:
            path: Points to training data in npz format
            batch_size: batch size
            data_col: data column name
            label_col: label column name

        Returns
        -------
        dict[str, tf.data.Dataset|int]
            a dictionary with dataset_train, dataset_val, dataset_total, step_train, step_val and step_total
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            data_total: Union[Dict[str, np.ndarray], np.ndarray] = data[f'{data_col}_total']
            label_total: Union[Dict[str, np.ndarray], np.ndarray] = data[f'{label_col}_total']

            data_train: Union[Dict[str, np.ndarray], np.ndarray] = data[f'{data_col}_train']
            label_train: Union[Dict[str, np.ndarray], np.ndarray] = data[f'{label_col}_train']

            data_val: Union[Dict[str, np.ndarray], np.ndarray] = data[f'{data_col}_val']
            label_val: Union[Dict[str, np.ndarray], np.ndarray] = data[f'{label_col}_val']

        if isinstance(data_total, dict):
            # multi input
            len_total = list(data_total.values())[0].shape[0]
            len_train = list(data_train.values())[0].shape[0]
            len_val = list(data_val.values())[0].shape[0]
        else:
            # single input
            len_total = data_total.shape[0]
            len_train = data_train.shape[0]
            len_val = data_val.shape[0]

        dataset_train = tf.data.Dataset.from_tensor_slices((data_train, label_train)).shuffle(len_train)\
            .batch(batch_size)
        dataset_val = tf.data.Dataset.from_tensor_slices((data_val, label_val)).batch(batch_size)
        dataset_total = tf.data.Dataset.from_tensor_slices((data_total, label_total)).shuffle(len_total)\
            .batch(batch_size)

        step_train = int(np.ceil(len_train / batch_size))
        step_val = int(np.ceil(len_val / batch_size))
        step_total = int(np.ceil(len_total / batch_size))

        data = {
            'dataset_train': dataset_train,
            'dataset_val': dataset_val,
            'dataset_total': dataset_total,
            'step_train': step_train,
            'step_val': step_val,
            'step_total': step_total,
        }
        return data

    def save(self, path: Union[str, Path]) -> None:
        """
        save model

        Parameters
        ----------
        path : str, Path
            The model home path
        """
        path = str(Path(path) / f'{self.model.name}.h5')
        self.model.save(path)

    def load(self, path: Union[str, Path]) -> None:
        """
        load model

        Parameters
        ----------
        path : str, Path
            The model home path
        """
        path = str(Path(path) / f'{self.model.name}.h5')
        self.model = tf.keras.models.load_model(path)


class FernTrainer(FernBaseTrainer):
    def __init__(
            self,
            model: FernModel,
            path_data: Union[str, Path],
            opt: str = 'adam',
            lr: float = 0.003,
            batch_size: int = 64,
            data_col: str = 'data',
            label_col: str = 'label'):
        """
        model trainer

        This can be used in one output and multi outputs

        Args:
            model: the model to be trained
            path_data: Points to training data in npz format
            opt: the name of optimizer
            lr: learning rate
            batch_size: batch size
            data_col: data column name
            label_col: label column name
        """
        self.model = model
        self.optimizer = tf.keras.optimizers.get({
            'class_name': opt,
            'config': {'learning_rate': lr}
        })

        self.metrics, self.early_stop_metric = self.setup_metrics()

        self.data = self.load_data(path_data, batch_size, data_col, label_col)

    def train(self, epochs: int = 1, early_stop: Optional[int] = None, mode: str = 'search') -> Tuple[float, int]:
        """
        train the model

        Args:
            epochs: epoch number
            early_stop:
                If None, no early stop will be used.
                If is number, after number times no update, will stop training early.
            mode:
                search: train model with data_train
                server: train model with data_train and data_val

        Returns:
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

            for data, label in ProgressBar(dataset, desc='Train: ', total=total):
                self.train_step(data, label)

            for data, label in ProgressBar(self.data['dataset_val'], desc='Val: ', total=self.data['step_val']):
                self.val_step(data, label)

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
            logger.info(log)

            if stop_flag:
                break
        return best_score, best_epoch

    @tf.function
    def train_step(
            self,
            data: Union[Dict[str, tf.Tensor], tf.Tensor],
            label: Union[Dict[str, tf.Tensor], tf.Tensor]
    ) -> None:
        """
        train one step

        Args:
            data: the input train data
            label: the output train labels
        """
        with tf.GradientTape() as tape:
            prediction = self.model(data)
            loss = self.loss(label, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        acc = self.acc(label, prediction)

        self.metrics['train']['loss'].update_state(loss)
        self.metrics['train']['acc'].update_state(acc)

    @tf.function
    def val_step(
            self,
            data: Union[Dict[str, tf.Tensor], tf.Tensor],
            label: Union[Dict[str, tf.Tensor], tf.Tensor]
    ) -> None:
        """
        val one step

        Args:
            data: the input train data
            label: the output train labels
        """
        prediction = self.model(data)
        loss = self.loss(label, prediction)
        acc = self.acc(label, prediction)

        self.metrics['val']['loss'].update_state(loss)
        self.metrics['val']['acc'].update_state(acc)

    @staticmethod
    def setup_metrics() -> Tuple[Dict[str, Dict[str, Metric]], Metric]:
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

    @staticmethod
    def loss(ys_desired: Union[Dict[str, tf.Tensor], tf.Tensor], ys_predicted: Union[Dict[str, tf.Tensor], tf.Tensor])\
            -> tf.Tensor:
        """
        loss function

        suitable for:
            1. single label (If it is 2 category case, then you also need to convert to one-hot format)
            2. single output (array), multiple output (dict[str, array])
        not suitable for:
            1. multiple label (you should change the loss function)

        Args:
            ys_desired: each with shape (None, m)
            ys_predicted: each with shape (None, m)

        Returns:
            with shape ()
        """
        assert type(ys_desired) is type(ys_predicted), 'Please make sure ys_desired and ys_predicted have same type'
        if isinstance(ys_desired, tf.Tensor):
            ys_predicted = {'label': ys_predicted}
            ys_desired = {'label': ys_desired}

        sum_res = 0
        for key in ys_desired:
            # if use multi label output, change to tf.losses.binary_crossentropy
            res = tf.losses.categorical_crossentropy(ys_desired[key], ys_predicted[key])
            sum_res += res
        sum_res = tf.reduce_mean(sum_res)
        return sum_res

    @staticmethod
    def acc(ys_desired: Union[Dict[str, tf.Tensor], tf.Tensor], ys_predicted: Union[Dict[str, tf.Tensor], tf.Tensor])\
            -> tf.Tensor:
        """
        accuracy function

        This function is for one output, multi outputs and for one-hot prediction.
        If you want to use multi-label prediction, you should rewrite this function.

        Args:
            ys_desired: each with shape (None, m)
            ys_predicted: each with shape (None, m)

        Returns:
            with shape (None, 1). When multi outputs, only while all outputs is True, the response will be True
        """
        assert type(ys_desired) is type(ys_predicted), 'Please make sure ys_desired and ys_predicted have same type'
        if isinstance(ys_desired, tf.Tensor):
            ys_predicted = {'label': ys_predicted}
            ys_desired = {'label': ys_desired}

        res = tf.constant(1)
        for key in ys_desired:
            # if use multi label output, you should change this code block
            y_true = tf.argmax(ys_desired[key], axis=-1)
            y_pred = tf.argmax(ys_predicted[key], axis=-1)
            res *= tf.cast(tf.equal(y_true, y_pred), tf.float32)
        return res


def compile_keras_model(
        model: tf.keras.Model,
        opt: str = 'adam',
        lr: float = 0.003,
        run_eagerly: Optional[int] = None,
        multi_classification: bool = True,
) -> tf.keras.Model:
    """
    编译keras模型

    Args:
        model: keras model
        opt: 优化器名字
        lr: 学习率
        run_eagerly: 如果是1, 用于动态图调试
        multi_classification: 如果是多标签的, 那么使用 binary_crossentropy; 否则使用

    Returns:
        编译后的模型
    """
    optimizer = tf.keras.optimizers.get({
        'class_name': opt,
        'config': {'learning_rate': lr}
    })

    if multi_classification:
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.binary_crossentropy,
            metrics=[
                tf.keras.metrics.binary_accuracy,
                tf.keras.metrics.AUC(),
            ],
            run_eagerly=run_eagerly,
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=[
                tf.keras.metrics.categorical_accuracy,
                tf.keras.metrics.AUC(),
            ],
            run_eagerly=run_eagerly,
        )
    return model


class FernKerasModelTrainer(object):
    def __init__(self,
                 model: tf.keras.Model,
                 ckpt_home: str,
                 log_home: str,
                 train_ds: tf.data.Dataset,
                 val_ds: Optional[tf.data.Dataset] = None,
                 version: int = 1):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.log_dir = Path(log_home) / f'{model.name}' / str(version)
        self.ckpt_dir = Path(ckpt_home) / f'{model.name}' / str(version) / 'weight.{epoch:02d}'

    def train(self, epochs: int = 1, steps_per_epoch: Optional[int] = None, verbose=1) -> tf.keras.callbacks.History:
        callbacks = [
            tf.keras.callbacks.TensorBoard(self.log_dir, update_freq=100, write_images=True, embeddings_freq=1),
            tf.keras.callbacks.ModelCheckpoint(self.ckpt_dir)
        ]
        kwargs = dict(
            x=self.train_ds,
            callbacks=callbacks,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=verbose
        )

        if self.val_ds:
            history = self.model.fit(validation_data=self.val_ds, **kwargs)
        else:
            history = self.model.fit(**kwargs)
        return history
