import logging
import os

from .benchmark import (
    setup_environment,
    run_benchmark,
    get_default_prof_config,
    run_full_benchmark
)

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
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 显式设置所有 utils 子模块的 logger 级别
    for module_name in ['utils.benchmark', 'utils.profiler', 'utils.judge']:
        logger = logging.getLogger(module_name)
        logger.setLevel(numeric_level)
        logger.propagate = True

__all__ = [
    'setup_environment',
    'run_benchmark',
    'get_default_prof_config',
    'run_full_benchmark',
    'setup_logging'
]
