# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""common function"""
import logging
import pathlib
import subprocess
from typing import *

import tensorflow as tf
from tqdm import tqdm
import yaml


logger = logging.getLogger('Fern')


def check_path(path):
    """
    check if path exits. If not exit, the path.parent will be created.

    Parameters
    ----------
    path : str, Path
        path to be check
    """
    path = pathlib.Path(path).parent
    if not path.exists():
        path.mkdir(parents=True)


def get_available_gpu(min_memory=0):
    """
    find the gpu of which free memory is largest

    References
    ----------
    refer to link: https://stackoverflow.com/a/59571639

    Parameters
    ----------
    min_memory : int
        Minimum allowable memory in MB

    Returns
    -------
    tuple[int, int] or None
        - If there is gpu available, return (gpu index, free memory)
        - Else return (None, 0)
    """
    command = "nvidia-smi --query-gpu=memory.free --format=csv"  # output: b'memory.free [MiB]\n48600 MiB\n48274 MiB\n'
    try:
        memory_free_info = subprocess.check_output(command.split())
    except FileNotFoundError as e:
        return None, 0
    memory_info = memory_free_info.decode('ascii').split('\n')[1:-1]
    res = []
    for i, memory in enumerate(memory_info):
        memory = int(memory.split()[0])
        if memory > min_memory:
            res.append((i, memory))
    res = sorted(res, key=lambda item: item[1], reverse=True)
    if res:
        return res[0]
    else:
        return None, 0


def set_gpu(index: Optional[int] = None, growth: bool = True):
    """
    set which GPU to use

    Args:
        index: the gpu index
        growth: whether to limit gpu memory growth
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if growth:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    if gpus and isinstance(index, int):
        try:
            tf.config.experimental.set_visible_devices(gpus[index], 'GPU')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logger.info(e)
        except IndexError as e:
            # there is no such a gpu found
            logger.info(e)


class ProgressBar(tqdm):
    def __init__(self, *arg, **kwargs):
        super().__init__(ascii='->', leave=False, *arg, **kwargs)


def read_config(path: str):
    """读取yaml配置文件"""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data

