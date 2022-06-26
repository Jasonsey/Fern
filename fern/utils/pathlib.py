# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""
tools for files in hdfs and local file system
"""
from typing import *
import os
import re
import pathlib
import subprocess

from fern.logging import Logging


def is_hdfs_path(path: str):
    """检查path是否是合法的HDFS path, 合法的hdfs path格式: hdfs://a/..."""
    paths = re.split(r'(/+)', path)
    if ('hdfs' not in path or '..' in path or './' in path) or (len(paths) < 4):
        ret = False
    else:
        ret = True
    return ret


class HDFSPurePath(object):
    """只做HDFS路径操作"""
    def __init__(self, path: str):
        path = re.sub(r'/+$', '', path)
        self._path = path
        self._root = self._get_root(path)
        self._relative_path = self._get_relative_path(path)
        self.logger = Logging()

    def __new__(cls, path, *args, **kwargs):
        if isinstance(path, HDFSPurePath):
            self = path
        else:
            self = object.__new__(cls)
            self = cls.__init__(self, path)
        return self

    @classmethod
    def _make_path(cls, root: str = None, relative_path: str = None, abspath: str = None):
        """根据参数创建新的path

        Args:
            root: format 'hdfs://xxx'
            relative_path: format 'a/b' or '.'
            abspath: format 'hdfs://xx/a/b'
        
        Returns:
            PurePath
        """
        if abspath is None:
            if relative_path == '.':
                abspath = root
            else:
                abspath = root + '/' + relative_path
        return cls(abspath)

    @staticmethod
    def _get_root(path) -> str:
        """读取path的root目录: hdfs://a/b/c -> hdfs://a"""
        ret = re.findall(r'^hdfs://[^/]+', path)
        assert len(ret) == 1, f'Unable to find the root path from: {path}'
        ret = ret[0]
        return ret

    def _get_relative_path(self, path) -> str:
        """读取hdfs path除了根目录外的相对路径, 用于所有的路径操作

        Returns:
            仅返回如下格式: ['a/b', '.']
        """
        root = self._get_root(path)
        path = re.sub(rf'^{root}', r'.', path)
        path = pathlib.PosixPath(path)
        path = str(path)
        return path

    @property
    def root(self):
        return self._root

    @property
    def parent(self):
        """获取当前路径的父路径"""
        relative_path = str(pathlib.PosixPath(self._relative_path).parent)
        return self._make_path(root=self._root, relative_path=relative_path)
    
    @property
    def name(self) -> str:
        """a/b.c -> b.c"""
        return pathlib.PosixPath(self._relative_path).name

    @property
    def stem(self) -> str:
        """a/b.c -> b"""
        return pathlib.PosixPath(self._relative_path).stem
    
    @property
    def suffix(self) -> str:
        """a/b.c -> .c"""
        return pathlib.PosixPath(self._relative_path).suffix

    def __repr__(self):
        """打印路径文本: for str function and print function"""
        relative_path = self._relative_path
        ret = [self._root]
        if relative_path != '.':
            ret.append(relative_path)
        ret = '/'.join(ret)
        return ret
    
    def __truediv__(self, other):
        """hdfs://a / b => hdfs://a/b"""
        relative_path = str(pathlib.PosixPath(self._relative_path) / str(other))
        return self._make_path(root=self._root, relative_path=relative_path)
    
    def __eq__(self, other):
        """a == b"""
        assert isinstance(other, HDFSPurePath)
        return str(self) == str(other)


class HDFSPath(HDFSPurePath):
    """实现HDFS文件常用查询"""
    def exists(self) -> bool:
        ret = False
        with subprocess.Popen(['hdfs', 'dfs', '-ls', str(self)], stderr=subprocess.PIPE).stderr as oup:
            for line in oup:
                line = line.decode('utf-8')
                if 'No such file or directory' in line:
                    ret = True
                    break
        return ret

    def is_dir(self) -> bool:
        return self.exists() and not self.is_file()

    def is_file(self) -> bool:
        if not self.exists():
            return False
        ret = False
        if self.parent != self:
            # not root dir
            name = self.name
            with subprocess.Popen(['hdfs', 'dfs', '-ls', str(self.parent)], stdout=subprocess.PIPE).stdout as oup:
                for line in oup:
                    # line: b'-rw-r--r-- 3 hive supergroup 0 2022-05-18 06:35
                    # hdfs://ana-hdfs/tmp/ai_tech/max/sql_test/_SUCCESS\n'
                    line = line.decode('utf-8')
                    if 'hdfs' in line and name in line:
                        # current file or folder
                        path_type = line.strip().split()[0]
                        ret = path_type[0] != 'd'  # drw-r--r--
                        break
        return ret

    def glob(self, pattern: str) -> Generator:
        """递归当前目录以及其子目录， 当前目录必须是文件夹

        Args:
            pattern: 支持的pattern模式， 和pathlib中的匹配规则一致
        """
        with subprocess.Popen(['hdfs', 'dfs', '-ls', f'{self._path}/{pattern}'], stdout=subprocess.PIPE).stdout as oup:
            for line in oup:
                line = line.decode('utf-8')
                if 'hdfs' in line:
                    path = line.strip().split()[-1]
                    path = self._make_path(abspath=path)
                    yield path


class Path(HDFSPath):
    def __new__(cls, path: str):
        if is_hdfs_path(path):
            self = object.__new__(cls)
            cls.__init__(self, path)
        else:
            self = pathlib.Path(path)
        return self
