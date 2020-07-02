# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""common function"""
import pathlib

import nltk
import jieba
from nltk import tokenize
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


def read_words(words_path):
    """
    read user words, stop words and word library from path

    Parameters
    ----------
    words_path : str, Path, None
        words path

    Returns
    -------
    list[str]
        user word list and stop word list
    """
    def read(path):
        with open(path, mode='r', encoding='utf-8') as f:
            res = f.readlines()
        res = [item.strip().lower() for item in res]
        res = list(set(res))
        return res

    if words_path is None or not pathlib.Path(words_path).exists():
        words = []
    else:
        words = read(words_path)
    return words


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


class Sequence2Words(object):
    def __init__(self, language='en', user_words=None):
        """
        split sequence into word list

        Parameters
        ----------
        language : str
            what the precessing language is
        user_words : str, pathlib.Path, optional
            make sure a word in one line
        """
        if language == 'en':
            nltk.download('punkt')
            user_words = read_words(user_words)
            user_words = [tuple(item.split(' ')) for item in user_words]
            tokenizer = tokenize.MWETokenizer(user_words, separator=' ')
            self.cut_func = lambda string: tokenizer.tokenize(nltk.word_tokenize(string))
        elif language == 'zh':
            if user_words:
                jieba.load_userdict(user_words)
            self.cut_func = jieba.lcut
        else:
            raise ValueError(f'Not support for language: {language}')

    def __call__(self, text):
        return self.cut_func(text)
