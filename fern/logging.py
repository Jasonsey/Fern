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
    配置日志类

    说明
    ====

    logging handler继承
    ----
    logging支持继承的方式, 例如::

        logger_root = logging.getLogger('root') # or logger_root = logging.getLogger()
        logger_a = logging.getLogger('a')
        logger_b = logging.getLogger('a.b')

    在 logger_a 中添加handler_1, 在logger_b中添加 handler_2;
    执行 logger_b.info('xxx'), 那么它的处理逻辑是:

    - 确认logger_b的level, 如果info < level, 那么下面的步骤都被忽略
    - 确认logger_b各个handler的level, 如果 info < level, 不做跳过这个handler
    - 搜索logger_b中的handler并执行
    - 搜索logger_a中的handler并执行
    - 搜索logger_root中的handler并执行
    - 如果前面几个步骤都没有找到handler, 那么会使用 "临时" 创建一个 StreamHandler 并执行

    logging handler 说明
    ----
    logging handler 在被创建时, 默认的level值=0, 例如::

        h = logging.StreamHandler()
        h.level    # 0

    也就是说, 默认情况下, 当前handler不会影响logger的输出

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

    def __init__(self, name=__name__, level='info'):
        logger = logging.getLogger(name)
        if level.upper() in self.levels:
            logger.setLevel(self.levels[level.upper()])
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
