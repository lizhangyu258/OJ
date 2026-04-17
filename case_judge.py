import os
import json
import glob
import importlib.util
import logging

from utils.judge import build_serializable_benchmark_result
from utils.judge import build_empty_result
from utils.judge import generate_final_result
from utils.judge import load_baseline_data

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# 测试用例目录
TESTCASES_DIR = os.path.join(ROOT_DIR, 'testcases')
# Baseline数据目录
BASELINE_DIR = os.path.join(ROOT_DIR, 'baseline')
BASELINE_DATA_FILE = os.path.join(BASELINE_DIR, 'data.yaml')

# 获取所有测试用例文件
def get_testcase_files():
    testcase_pattern = os.path.join(TESTCASES_DIR, '*.py')
    testcase_files = glob.glob(testcase_pattern)
    # 按文件名排序
    testcase_files.sort()
    logger.info(f"Found {len(testcase_files)} test cases: {[os.path.basename(f) for f in testcase_files]}")
    return testcase_files


def load_testcase_module(testcase_file):
    module_name = f"testcase_{os.path.splitext(os.path.basename(testcase_file))[0]}"
    spec = importlib.util.spec_from_file_location(module_name, testcase_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load testcase module from {testcase_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 运行单个测试用例
def run_testcase(testcase_file):
    testcase_name = os.path.basename(testcase_file)
    logger.info(f"Running test case: {testcase_name}")

    try:
        module = load_testcase_module(testcase_file)
        if not hasattr(module, 'main'):
            raise AttributeError(f"Testcase {testcase_name} does not define main()")

        benchmark_result = build_serializable_benchmark_result(module.main())
        if benchmark_result is None:
            raise ValueError(f"Testcase {testcase_name} returned no benchmark result")

        result = {
            'testcase': testcase_name,
            'exit_code': 0,
            'benchmark_result': benchmark_result,
            'error_message': '',
        }
        logger.info(f"Test case {testcase_name} completed successfully")
        return result
    except Exception as e:
        logger.exception(f"Error running test case {testcase_name}")
        return {
            'testcase': testcase_name,
            'exit_code': 1,
            'benchmark_result': None,
            'error_message': str(e),
        }


def main():
    logger.info("Starting case evaluation...")
    
    # 获取所有测试用例
    testcase_files = get_testcase_files()
    
    if not testcase_files:
        logger.warning("No test cases found in testcases directory")
        final_result = build_empty_result()
        return
    
    # 运行所有测试用例
    testcase_results = []
    for testcase_file in testcase_files:
        result = run_testcase(testcase_file)
        testcase_results.append(result)
    
    # 生成最终结果
    baseline_data = load_baseline_data(BASELINE_DATA_FILE)
    final_result = generate_final_result(
        testcase_results,
        baseline_data
    )
    
    logger.info(f"final_result json: {json.dumps(final_result, indent=2)}")
    logger.info("Case evaluation completed")

if __name__ == '__main__':
    main()
