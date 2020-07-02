# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""common function"""
import pathlib
from tqdm import tqdm


def read_library_size(path):
    """
    read the length of the word/label library
    this will skip the space line automatically

    Parameters
    ----------
    path : str, pathlib.Path
        word library path

    Returns
    -------
    int
        length of the word library
    """
    res = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                res.append(line)
    return len(res)


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


class ProgressBar(tqdm):
    def __init__(self, *arg, **kwargs):
        super().__init__(ascii='->', leave=False, *arg, **kwargs)
