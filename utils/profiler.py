import os
import logging
import csv
import shutil
import math
import statistics
from typing import Iterable, Optional, Tuple

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


def get_default_traced_graph_cache_dir() -> str:
    """
    获取默认 traced_graph_cache 目录路径（项目根目录下的 traced_graph_cache 目录）
    """
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(utils_dir)
    return os.path.join(project_root, "traced_graph_cache")


def get_default_cleanup_dirs() -> Tuple[str, ...]:
    """
    获取默认需要清理的目录列表。
    """
    return (
        get_default_prof_dir(),
        get_default_traced_graph_cache_dir(),
    )


def cleanup_directories(
    target_dirs: Optional[Iterable[str]] = None,
    print_log: bool = True
) -> int:
    """
    删除运行评测生成的目录产物。

    Args:
        target_dirs: 要删除的目录列表；不传时清理默认产物目录
        print_log: 是否打印日志

    Returns:
        实际删除的目录数量
    """
    if target_dirs is None:
        target_dirs = get_default_cleanup_dirs()

    removed_count = 0
    for target_dir in target_dirs:
        if not os.path.exists(target_dir):
            if print_log:
                logger.info(f"Directory does not exist, skip cleanup: {target_dir}")
            continue
        if not os.path.isdir(target_dir):
            raise NotADirectoryError(f"Cleanup target is not a directory: {target_dir}")

        shutil.rmtree(target_dir)
        removed_count += 1
        if print_log:
            logger.info(f"Removed directory: {target_dir}")
    return removed_count


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


def _filter_outlier_step_times(
    step_times,
    max_outliers: int = 10,
    min_samples: int = 10,
    modified_z_threshold: float = 3.5,
):
    if len(step_times) <= min_samples or max_outliers <= 0:
        return step_times, 0

    median_time = statistics.median(step_times)
    deviations = [abs(step_time - median_time) for step_time in step_times]
    mad = statistics.median(deviations)
    max_remove_count = min(max_outliers, len(step_times) - min_samples)
    if max_remove_count <= 0:
        return step_times, 0

    if mad > 0:
        outlier_candidates = []
        for index, step_time in enumerate(step_times):
            modified_z_score = 0.6745 * abs(step_time - median_time) / mad
            if modified_z_score > modified_z_threshold:
                outlier_candidates.append((modified_z_score, index))
    else:
        relative_threshold = abs(median_time) * 0.05
        absolute_threshold = max(relative_threshold, 1e-12)
        outlier_candidates = [
            (abs(step_time - median_time), index)
            for index, step_time in enumerate(step_times)
            if abs(step_time - median_time) > absolute_threshold
        ]

    if not outlier_candidates:
        return step_times, 0

    outlier_candidates.sort(reverse=True)
    removed_indices = {
        index for _, index in outlier_candidates[:max_remove_count]
    }
    filtered_step_times = [
        step_time
        for index, step_time in enumerate(step_times)
        if index not in removed_indices
    ]
    return filtered_step_times, len(removed_indices)


def get_step_time_from_csv(csv_path: str) -> Optional[float]:
    """
    从step_trace_time.csv文件中读取第3列（索引2）的step算子计算时间，剔除异常值后计算平均执行时间
    
    Args:
        csv_path: step_trace_time.csv文件的完整路径
    
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
            
            step_times = []
            
            for i, values in enumerate(rows[1:], start=2):
                if len(values) < 3:
                    logger.warning(f"CSV row {i} has less than 3 columns, skipping")
                    continue
                
                step_time_str = values[2]
                
                try:
                    step_time = float(step_time_str)
                    if not math.isfinite(step_time):
                        logger.warning(f"Non-finite step time '{step_time_str}' in row {i}, skipping")
                        continue
                    step_times.append(step_time)
                except ValueError:
                    logger.warning(f"Cannot convert '{step_time_str}' to float in row {i}, skipping")
                    continue
            
            if not step_times:
                logger.warning(f"No valid data rows found in CSV: {csv_path}")
                return None
            
            filtered_step_times, removed_count = _filter_outlier_step_times(step_times)
            total_time = sum(filtered_step_times)
            avg_time = total_time / len(filtered_step_times)
            logger.info(
                "Average step execution time: %s "
                "(used %s/%s rows after outlier filter, removed: %s, sum: %s)",
                avg_time,
                len(filtered_step_times),
                len(step_times),
                removed_count,
                total_time,
            )
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
