from typing import *
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric

from .model import FernModel


class FernTrainer:
    model: FernModel
    optimizer: Optional[tf.optimizers.Optimizer]
    data: Dict[str, Union[tf.data.Dataset, int]]
    metrics: Dict[str, Dict[str, Metric]]
    early_stop_metric: Metric
    label_weight: Union[Dict[str, np.ndarray], np.ndarray]

    def __init__(self, model: FernModel, path_data: Union[str, Path], opt: str='adam', lr: float=0.003, batch_size: int=64, data_col: str='data', label_col: str='label') -> None: ...

    def train(self, epochs: int=1, early_stop: Optional[int]=None, mode: str='search') -> Tuple[float, int]: ...

    def train_step(self, data: Union[Dict[tf.Tensor], tf.Tensor], label: Union[Dict[tf.Tensor], tf.Tensor]) -> None: ...

    def val_step(self, data: Union[Dict[tf.Tensor], tf.Tensor], label: Union[Dict[tf.Tensor], tf.Tensor]) -> None: ...

    @staticmethod
    def setup_metrics() -> Tuple[dict[str, dict[str, Metric]], Metric]: ...

    @staticmethod
    def loss(ys_desired: Union[Dict[str, tf.Tensor], tf.Tensor], ys_predicted: Union[Dict[str, tf.Tensor], tf.Tensor]) -> tf.Tensor: ...

    @staticmethod
    def acc(ys_desired: Union[Dict[str, tf.Tensor], tf.Tensor], ys_predicted: Union[Dict[str, tf.Tensor], tf.Tensor]) -> tf.Tensor: ...

    @staticmethod
    def load_data(path: Union[str, Path], batch_size: int, data_col: str, label_col: str) -> Dict[str, Union[tf.data.Dataset, int]]: ...

    def save(self, path: Union[str, Path]) -> None: ...

    def load(self, path: Union[str, Path]) -> None: ...
