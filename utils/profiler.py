import os
import logging
import uuid
import csv
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def get_default_prof_dir() -> str:
    """
    获取默认的prof目录路径（项目根目录下的prof目录）
    
    Returns:
        prof目录的绝对路径
    """
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(utils_dir)
    return os.path.join(project_root, "prof")


def generate_unique_id() -> str:
    """生成全局唯一ID"""
    return str(uuid.uuid4())


def create_prof_output_dir(base_dir: str, unique_id: Optional[str] = None) -> Tuple[str, str]:
    """
    创建profiler输出目录，使用唯一ID作为子目录名
    
    Args:
        base_dir: 基础输出目录（通常是项目根目录下的prof目录）
        unique_id: 可选的唯一ID，如果不提供则自动生成
    
    Returns:
        Tuple[完整输出路径, 唯一ID]
    """
    if unique_id is None:
        unique_id = generate_unique_id()
    
    output_dir = os.path.join(base_dir, unique_id)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir, unique_id


def get_avg_execution_time_from_csv(csv_path: str) -> Optional[float]:
    """
    从op_statistic.csv文件中读取平均执行时间（倒数第三列）
    
    Args:
        csv_path: op_statistic.csv文件的完整路径
    
    Returns:
        平均执行时间（float），如果读取失败则返回None
    """
    if not os.path.exists(csv_path):
        logger.warning(f"CSV file not found: {csv_path}")
        return None
    
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            if len(rows) < 2:
                logger.warning(f"CSV file has less than 2 rows: {csv_path}")
                return None
            
            values = rows[1]
            
            if len(values) < 3:
                logger.warning(f"CSV file has less than 3 columns: {csv_path}")
                return None
            
            avg_time_str = values[-3]
            
            try:
                avg_time = float(avg_time_str)
                logger.info(f"Average execution time: {avg_time}")
                return avg_time
            except ValueError:
                logger.warning(f"Cannot convert '{avg_time_str}' to float")
                return None
                
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {e}")
        return None


def find_and_parse_op_statistic(prof_output_dir: str) -> Optional[float]:
    """
    在prof输出目录中查找op_statistic.csv并解析平均执行时间
    
    Args:
        prof_output_dir: profiler输出目录
    
    Returns:
        平均执行时间（float），如果找不到或解析失败则返回None
    """
    csv_filename = "op_statistic.csv"
    csv_path = os.path.join(prof_output_dir, csv_filename)
    
    if not os.path.exists(csv_path):
        for root, dirs, files in os.walk(prof_output_dir):
            if csv_filename in files:
                csv_path = os.path.join(root, csv_filename)
                break
        else:
            logger.warning(f"{csv_filename} not found in {prof_output_dir}")
            return None
    
    logger.info(f"Found op_statistic.csv at: {csv_path}")
    return get_avg_execution_time_from_csv(csv_path)


def setup_profiler_output(base_prof_dir: str, print_log: bool = True) -> str:
    """
    设置profiler输出目录的完整流程
    
    Args:
        base_prof_dir: 基础prof目录（项目根目录下的prof目录）
        print_log: 是否打印日志
    
    Returns:
        完整输出路径
    """
    os.makedirs(base_prof_dir, exist_ok=True)
    
    output_dir, unique_id = create_prof_output_dir(base_prof_dir)
    
    if print_log:
        logger.info(f"Profiler output directory: {output_dir}")
        logger.info(f"Unique ID: {unique_id}")
    
    return output_dir
