# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""test pathlib"""
import pathlib

from fern.utils.pathlib import Path, HDFSPurePath


class TestPath(object):
    @classmethod
    def setup_class(cls):
        cls.path1 = Path('/a/b.c')
        cls.path2 = Path('hdfs://a/b.c')

    def test_path_type(self):
        assert isinstance(self.path1, pathlib.Path)
        assert isinstance(self.path2, HDFSPurePath)

    def test_path_print(self):
        assert str(self.path1) == '/a/b.c'
        assert str(self.path2) == 'hdfs://a/b.c'

    def test_path_operator(self):
        assert str(self.path1 / 'x') == '/a/b.c/x'
        assert str(self.path2 / 'x') == 'hdfs://a/b.c/x'

    def test_path_root(self):
        assert self.path1.root == '/'
        assert self.path2.root == 'hdfs://a'

    def test_parent(self):
        assert self.path1.parent == Path('/a')
        assert self.path2.parent == Path('hdfs://a')

    def test_name(self):
        assert self.path1.name == 'b.c'
        assert self.path1.name == 'b.c'

    def test_stem(self):
        assert self.path1.stem == 'b'
        assert self.path2.stem == 'b'

    def test_suffix(self):
        assert self.path1.suffix == '.c'
        assert self.path2.suffix == '.c'
