# Fern
#
# Author: Jasonsey
# Email: 2627866800@qq.com
#
# =============================================================================
"""common function"""
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


class ProgressBar(tqdm):
    def __init__(self, *arg, **kwargs):
        super().__init__(ascii='->', leave=False, *arg, **kwargs)
