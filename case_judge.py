import os
import subprocess
import json
import glob
import logging
import argparse

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
# 输出目录
OUTPUTS_DIR = os.path.join(ROOT_DIR, 'outputs')
# Baseline数据目录
BASELINE_DIR = os.path.join(ROOT_DIR, 'baseline')
BASELINE_DATA_FILE = os.path.join(BASELINE_DIR, 'data.yaml')

# 创建输出目录（如果不存在）
def create_output_dir():
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)
        logger.info(f"Created output directory: {OUTPUTS_DIR}")

# 获取所有测试用例文件
def get_testcase_files():
    testcase_pattern = os.path.join(TESTCASES_DIR, '*.py')
    testcase_files = glob.glob(testcase_pattern)
    # 按文件名排序
    testcase_files.sort()
    logger.info(f"Found {len(testcase_files)} test cases: {[os.path.basename(f) for f in testcase_files]}")
    return testcase_files

# 运行单个测试用例
def run_testcase(testcase_file):
    testcase_name = os.path.basename(testcase_file)
    output_file = os.path.join(OUTPUTS_DIR, f"{testcase_name}.out")
    error_file = os.path.join(OUTPUTS_DIR, f"{testcase_name}.err")
    
    logger.info(f"Running test case: {testcase_name}")
    
    try:
        # 运行测试用例脚本
        with open(output_file, 'w') as out, open(error_file, 'w') as err:
            process = subprocess.run(
                ['python3', testcase_file],
                stdout=out,
                stderr=err,
                timeout=300,  # 设置5分钟超时
                check=False
            )
        
        # 读取输出和错误
        with open(output_file, 'r') as f:
            stdout = f.read()
        
        with open(error_file, 'r') as f:
            stderr = f.read()
        
        result = {
            'testcase': testcase_name,
            'exit_code': process.returncode,
            'stdout': stdout,
            'stderr': stderr,
            'output_file': output_file,
            'error_file': error_file
        }
        
        if process.returncode == 0:
            logger.info(f"Test case {testcase_name} completed successfully")
        else:
            logger.warning(f"Test case {testcase_name} failed with exit code {process.returncode}")
        
        return result
        
    except subprocess.TimeoutExpired:
        logger.error(f"Test case {testcase_name} timed out")
        return {
            'testcase': testcase_name,
            'exit_code': -1,
            'stdout': '',
            'stderr': 'Timeout expired after 300 seconds',
            'output_file': output_file,
            'error_file': error_file
        }
    except Exception as e:
        logger.error(f"Error running test case {testcase_name}: {str(e)}")
        return {
            'testcase': testcase_name,
            'exit_code': -2,
            'stdout': '',
            'stderr': f'Unexpected error: {str(e)}',
            'output_file': output_file,
            'error_file': error_file
        }


def main():
    parser = argparse.ArgumentParser(description='OJ Case Judge')
    parser.add_argument('--use-baseline-eager', action='store_true', default=True,
                        help='Use eager as performance baseline')
    parser.add_argument('--use-baseline-compile', action='store_false', dest='use_baseline_eager',
                        help='Use compile as performance baseline')
    args = parser.parse_args()

    logger.info("Starting case evaluation...")
    
    # 创建输出目录
    create_output_dir()
    
    # 获取所有测试用例
    testcase_files = get_testcase_files()
    
    if not testcase_files:
        logger.warning("No test cases found in testcases directory")
        final_result = build_empty_result()
        print(json.dumps(final_result, indent=2))
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
        baseline_data,
        use_baseline_eager=args.use_baseline_eager
    )
    
    print(json.dumps(final_result, indent=2))
    
    logger.info("Case evaluation completed")

if __name__ == '__main__':
    main()
