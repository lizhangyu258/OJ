import logging
import os
import sys

def setup_logging(log_level=None):
    """
    统一设置项目的日志配置
    
    Args:
        log_level: 日志级别，可选 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
                   如果不提供，将从环境变量 LOG_LEVEL 读取，默认为 'INFO'
    """
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    numeric_level = getattr(logging, log_level, logging.INFO)
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    if root_logger.handlers:
        formatter = logging.Formatter(log_format)
        for handler in root_logger.handlers:
            handler.setLevel(numeric_level)
            handler.setFormatter(formatter)
    else:
        logging.basicConfig(
            level=numeric_level,
            format=log_format,
            stream=sys.stderr
        )
    
    # 显式设置项目 logger 级别
    for module_name in ['case_judge', 'utils.benchmark', 'utils.profiler', 'utils.judge']:
        logger = logging.getLogger(module_name)
        logger.setLevel(numeric_level)
        logger.propagate = True

__all__ = [
    'setup_logging'
]
