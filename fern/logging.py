# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""logging tool"""
from typing import *
import logging
from pathlib import Path


class Logging(object):
    """
    配置日志类, 如果是父类 `'.' not in name`, 那么允许继承, 不允许继承

    由于经常有程序修改 root logger 的配置(如tensorflow), 所有这里会默认屏蔽所有root logger的所有配置

    Examples
    ----
    - 在项目的根目录实例化logger::

        >>> from fern.logging import Logging
        >>> logger = Logging(__name__)
        >>> logger.add_stream_handler()

    - 在任何需要引用使用日志的地方执行::

        >>> import logging    # 注意是logging, 第一步已经把需要的信息注册到logging系统了, 可以无缝使用
        >>> logger = logging.getLogger(__name__)
    """

    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s: %(message)s')
    levels = dict(
        DEBUG=logging.DEBUG,
        INFO=logging.INFO,
        WARN=logging.WARN,
        WARNING=logging.WARN,
        ERROR=logging.ERROR
    )

    def __init__(self, name: str = 'Fern', level: Union[str, int] = 'info', propagate=False):
        logger = logging.getLogger(name)

        # 所有logger都不从root中继承handler, 但如果是子logger则要从父logger中继承
        if isinstance(name, str):
            logger.propagate = ('.' in name)
        # set level
        if isinstance(level, int):
            logger.setLevel(level)
        else:
            logger.setLevel(self.levels.get(str(level).upper(), 'WARN'))
        # add logger
        self.__logger = logger

    def __getattr__(self, item: str):
        if item in dir(self):
            # dir取出实例的所有属性
            return getattr(self, item)
        else:
            # 如果这个属性不在这个类里面, 那么就是__logger里面找, 用户logger.info的方案
            return getattr(self.__logger, item)

    def __repr__(self):
        return repr(self.__logger)

    def add_stream_handler(self, level: Union[str, int, None] = None, **kwargs):
        """
        日志导出到终端中, 新的stream handler会覆盖旧的

        Args:
            level: 默认的level是比 DEBUG 还小的 0
            **kwargs: other file handle config
        """
        stream_handler = logging.StreamHandler(**kwargs)
        stream_handler.setFormatter(fmt=self.formatter)

        if level in self.levels:
            level = self.levels[level]
        elif isinstance(level, int):
            level = level
        else:
            level = 0

        stream_handler.setLevel(level)

        # 避免重复添加handlers
        handlers = [handler for handler in self.__logger.handlers if not isinstance(handler, logging.StreamHandler)]
        handlers.append(stream_handler)
        self.__logger.handlers = handlers

    def add_file_handle(self, filename: Union[str, Path] = 'base.log', level: Union[str, int, None] = None, **kwargs):
        """
        日志导出到文件系统中, 新的file handler会覆盖旧的

        Args:
            filename: 文件路径
            level: 默认的level是比 DEBUG 还小的 0
            **kwargs: other file handle config
        """
        if not Path(filename).parent.exists():
            Path(filename).parent.mkdir(parents=True)

        file_handler = logging.FileHandler(filename=filename, **kwargs)
        file_handler.setFormatter(self.formatter)

        if level in self.levels:
            level = self.levels[level]
        elif isinstance(level, int):
            level = level
        else:
            level = 0

        file_handler.setLevel(level)

        # 避免重复添加handlers
        handlers = [handler for handler in self.__logger.handlers if not isinstance(handler, logging.FileHandler)]
        handlers.append(file_handler)
        self.__logger.handlers = handlers


class LoggingContext(object):
    """
    通过通过上下文, 执行前和执行结束打印结果

    Examples
    ----
    >>> with LoggingContext(print, 'test'):
    ...     print('xxx')
    Begin: test
    xxx
    End: test
    """
    def __init__(self, log_func: Callable, msg: str):
        """
        Args:
            log_func: logging.info等函数
            msg: 需要打印的数据
        """
        self.log_func = log_func
        self.msg = msg

    def __enter__(self):
        self.log_func(f'Begin: {self.msg}')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log_func(f'End: {self.msg}')
