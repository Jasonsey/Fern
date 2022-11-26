# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""data cleaning"""
from opencc import OpenCC


def full_width_to_half_width(string: str) -> str:
    """
    字符串全角转半角，例如：`１２３４５６７８９０ -> 1234567890`

    Args:
        string: 待转换字符串

    Returns:
        半角字符串
    """
    table = {ord(f): ord(t) for f, t in zip(
        u'＊・，。！？【】（）％＃＠＆１２３４５６７８９０、；—：■　ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ',
        u'*·,.!?[]()%#@&1234567890.;-:￭ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')}

    new_string = ''
    for char in string:
        char_ord = ord(char)
        if char_ord in table:
            char_ord = table[char_ord]
            char = chr(char_ord)
        new_string += char
    return new_string


def string_to_camel(string: str) -> str:
    """
    把下划线字符串转为驼峰模式
    """
    return ''.join([_string.capitalize() for _string in string.split('_')])


class TSConvertor(object):
    """繁体简体互相转换,

    References:
        https://github.com/yichen0831/opencc-python"""
    def __init__(self):
        self._s2t = OpenCC('s2t')
        self._t2s = OpenCC('t2s')

    def s2t(self, string: str) -> str:
        """简体转繁体"""
        string = self._s2t.convert(string)
        return string

    def t2s(self, string: str) -> str:
        """繁体转简体"""
        string = self._t2s.convert(string)
        return string
