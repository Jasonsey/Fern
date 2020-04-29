# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""logging tool"""
import logging
from pathlib import Path


class BaseLogging(object):
    """base logging class"""
    LEVELS = dict(
        DEBUG=logging.DEBUG,
        INFO=logging.INFO,
        WARN=logging.WARN,
        WARNING=logging.WARN,
        ERROR=logging.ERROR
    )

    def __init__(self, name='AI'):
        """
        base logging tools, set the global log level

        Parameters
        ----------
        name : str
            set logger name
        """
        self.__logger = logging.getLogger(name)
        self.set_level('debug')
        self.debug = self.__logger.debug
        self.info = self.__logger.info
        self.warn = self.__logger.warning
        self.warning = self.__logger.warning
        self.error = self.__logger.error

    def add_handler(self, handler):
        """
        add handler to logger

        Parameters
        ----------
        handler : logging.Handler
            a handler to be added into the logger
        """
        self.__logger.addHandler(handler)

    def set_level(self, level='info'):
        """
        set global level

        Parameters
        ----------
        level : str
            level should be in ['DEBUG', 'INFO', 'WARN', 'ERROR']
        """
        level = level.upper()
        assert level in self.LEVELS, f'the level should be in DEBUG, INFO, WARN, ERROR, but get {level}'
        self.__logger.setLevel(self.LEVELS[level])


class Logging(BaseLogging):
    """
    logging tools

    Examples
    ----------
    quick start

    >>> logger = Logging()
    >>> logger.info('log a info msg')

    set formatter before start

    >>> Logging.set_formatter(
    ...     fmt='%(asctime)s %(filename)s[line:%(lineno)d]%(levelname)s - %(message)s',
    ...     datefmt='%m/%d/%Y %I:%M:%S %p')
    >>> logger = Logging()
    >>> logger.warn('log a warn msg')
    """

    FORMATTER = logging.Formatter(fmt='%(asctime)s - %(name)s: %(message)s')

    @classmethod
    def set_formatter(cls, fmt=None, datefmt=None):
        """
        change the default output format

        Parameters
        ----------
        fmt : str, optional
            fmt
        datefmt : str, optional
            datefmt
        """
        cls.FORMATTER = logging.Formatter(fmt=fmt, datefmt=datefmt)

    def __init__(self, name='Fern', level='info'):
        """
        logging tools

        Parameters
        ----------
        name : str
            set logger name
        level : str
            logger level, for example, if set level = 'info', than only warn and error log will be printed
        """
        super().__init__(name)
        self.set_stream_handler(level=level)

    def set_stream_handler(self, level='warning'):
        """
        set stream handler

        Parameters
        ----------
        level : str
            should be in ['DEBUG', 'INFO', 'WARN', 'ERROR']
        """
        console = logging.StreamHandler()
        console.setFormatter(self.FORMATTER)
        console = self.set_handler_level(console, level)
        self.add_handler(console)

    def set_file_handle(self, filename='../logs/base.log', level='debug', **kwargs):
        """
        set file handle

        Parameters
        ----------
        filename : str, Path
            base log file name
        level : str
            should be in ['DEBUG', 'INFO', 'WARN', 'ERROR']
        kwargs : dict
            other file handle config
        """
        if not Path(filename).parent.exists():
            Path(filename).parent.mkdir(parents=True)

        file_logging = logging.FileHandler(filename=filename, **kwargs)
        file_logging.setFormatter(self.FORMATTER)
        file_logging = self.set_handler_level(file_logging, level)
        self.add_handler(file_logging)

    def set_handler_level(self, handler, level):
        """
        set handle level

        Parameters
        ----------
        handler : logging.Handler
            the handler to be set
        level : str
            should be in ['DEBUG', 'INFO', 'WARN', 'ERROR']

        Returns
        -------
        res : logging.Handler
            the handler
        """
        level = level.upper()
        assert level in self.LEVELS, f'the level should be in DEBUG, INFO, WARN, ERROR, but get {level}'
        handler.setLevel(self.LEVELS[level])
        return handler
