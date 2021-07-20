# Fern
#
# Author: Jason Lin
# Email: jason.m.lin@outlook.com
#
# =============================================================================
"""NLP text processing toolkit"""
from fern.logging import Logging

# 初始化一个logging工具, 用户当前模块日志打印
__logger = Logging()
__logger.add_stream_handler()
