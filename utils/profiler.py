import os
import logging
import csv
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def resolve_output_dir(base_dir: str, subdir: Optional[str] = None) -> str:
    """
    基于基础目录和可选子目录，得到最终保存目录。
    """
    if not subdir:
        return base_dir

    normalized_subdir = os.path.normpath(subdir.strip())
    if normalized_subdir in ("", "."):
        return base_dir
    if os.path.isabs(normalized_subdir) or normalized_subdir.startswith(".."):
        raise ValueError(f"Invalid output subdir: {subdir}")
    return os.path.join(base_dir, normalized_subdir)


def get_default_prof_dir() -> str:
    """
    获取默认的prof目录路径（项目根目录下的prof目录）
    
    Returns:
        prof目录的绝对路径
    """
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(utils_dir)
    return os.path.join(project_root, "prof")


def create_prof_output_dir(base_dir: str) -> Tuple[str, str]:
    """
    创建profiler输出目录
    
    Args:
        base_dir: 基础输出目录（通常是项目根目录下的prof目录）
    
    Returns:
        Tuple[完整输出路径, 唯一ID]
    """
    import uuid
    unique_id = str(uuid.uuid4())
    
    output_dir = os.path.join(base_dir, unique_id)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir, unique_id


def get_step_time_from_csv(csv_path: str) -> Optional[float]:
    """
    从op_statistic.csv文件中读取第3列（索引2）的step算子计算时间，计算平均执行时间
    
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
            
            total_time = 0.0
            valid_rows = 0
            
            for i, values in enumerate(rows[1:], start=2):
                if len(values) < 3:
                    logger.warning(f"CSV row {i} has less than 3 columns, skipping")
                    continue
                
                step_time_str = values[2]
                
                try:
                    step_time = float(step_time_str)
                    total_time += step_time
                    valid_rows += 1
                except ValueError:
                    logger.warning(f"Cannot convert '{step_time_str}' to float in row {i}, skipping")
                    continue
            
            if valid_rows == 0:
                logger.warning(f"No valid data rows found in CSV: {csv_path}")
                return None
            
            avg_time = total_time / valid_rows
            logger.info(f"Average step execution time: {avg_time} (sum of {valid_rows} rows: {total_time})")
            return avg_time
                
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {e}")
        return None


def find_and_parse_step_trace(prof_output_dir: str) -> Optional[float]:
    """
    在prof输出目录中查找step_trace_time.csv并解析平均执行时间
    
    Args:
        prof_output_dir: profiler输出目录
    
    Returns:
        平均执行时间（float），如果找不到或解析失败则返回None
    """
    csv_filename = "step_trace_time.csv"
    csv_path = os.path.join(prof_output_dir, csv_filename)
    
    if not os.path.exists(csv_path):
        for root, dirs, files in os.walk(prof_output_dir):
            if csv_filename in files:
                csv_path = os.path.join(root, csv_filename)
                break
        else:
            logger.warning(f"{csv_filename} not found in {prof_output_dir}")
            return None
    
    logger.info(f"Found step_trace_time.csv at: {csv_path}")
    return get_step_time_from_csv(csv_path)


def setup_profiler_output(
    base_prof_dir: str,
    artifact_subdir: Optional[str] = None,
    print_log: bool = True
) -> str:
    """
    设置profiler输出目录的完整流程
    
    Args:
        base_prof_dir: 基础prof目录（项目根目录下的prof目录）
        print_log: 是否打印日志
    
    Returns:
        完整输出路径
    """
    prof_dir = resolve_output_dir(base_prof_dir, artifact_subdir)
    os.makedirs(prof_dir, exist_ok=True)

    output_dir, unique_id = create_prof_output_dir(prof_dir)
    
    if print_log:
        logger.info(f"Profiler output directory: {output_dir}")
        logger.info(f"Unique ID: {unique_id}")
    
    return output_dir
