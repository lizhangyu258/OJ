import os
import subprocess
import json
import glob
import logging
import re
import yaml
from datetime import datetime

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

# 加载baseline数据
def load_baseline_data():
    """加载baseline数据"""
    if not os.path.exists(BASELINE_DATA_FILE):
        logger.warning(f"Baseline data file not found: {BASELINE_DATA_FILE}")
        return {}
    
    try:
        with open(BASELINE_DATA_FILE, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            logger.info(f"Loaded baseline data from {BASELINE_DATA_FILE}")
            return data if data else {}
    except Exception as e:
        logger.error(f"Error loading baseline data: {str(e)}")
        return {}

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

# 解析测试用例输出
def parse_testcase_output(testcase_result):
    """解析测试用例输出，提取性能指标等信息"""
    parsed_data = {
        'avg_execution_time': None
    }
    
    # 从输出中提取平均执行时间
    # 匹配格式: "Average execution time: {avg_exec_time} us"
    stdout = testcase_result['stdout']
    pattern = r'Average execution time:\s*([\d.]+)\s*us'
    match = re.search(pattern, stdout)
    
    if match:
        try:
            avg_time = float(match.group(1))
            parsed_data['avg_execution_time'] = avg_time
            logger.info(f"Parsed average execution time: {avg_time} us")
        except ValueError:
            logger.warning(f"Failed to parse average execution time from: {match.group(1)}")
    else:
        logger.warning("Average execution time not found in output")
    
    return parsed_data

# 计算测试用例评分
def calculate_testcase_score(testcase_result, parsed_output, baseline_data):
    """根据测试用例结果和解析的输出计算评分（加速比）"""
    testcase_name = testcase_result['testcase']
    
    # 基础分：测试用例成功运行
    if testcase_result['exit_code'] != 0:
        logger.error(f"Test case {testcase_name} failed with exit code {testcase_result['exit_code']}, score: 0.0")
        return 0.0
    
    # 获取当前测试用例的平均执行时间
    current_avg_time = parsed_output.get('avg_execution_time')
    if current_avg_time is None:
        logger.error(f"No average execution time found for {testcase_name}, score: 0.0")
        return 0.0
    
    # 获取baseline数据
    if testcase_name not in baseline_data:
        logger.error(f"No baseline data found for {testcase_name}, score: 0.0")
        return 0.0
    
    baseline_avg_time = baseline_data[testcase_name].get('avg_execution_time')
    if baseline_avg_time is None or baseline_avg_time <= 0:
        logger.error(f"Invalid baseline avg_execution_time for {testcase_name}, score: 0.0")
        return 0.0
    
    # 计算加速比 = baseline时间 / 当前时间
    speedup_ratio = baseline_avg_time / current_avg_time
    logger.info(f"Speedup ratio for {testcase_name}: {speedup_ratio:.4f} (baseline: {baseline_avg_time} us, current: {current_avg_time} us)")
    
    return speedup_ratio

# 生成最终结果
def generate_final_result(testcase_results):
    """根据所有测试用例的结果生成最终评测结果"""
    # 加载baseline数据
    baseline_data = load_baseline_data()
    
    # 检查所有测试用例是否成功
    all_passed = all(result['exit_code'] == 0 for result in testcase_results)
    verdict = 'AC' if all_passed else 'WA'
    
    # 计算总评分
    total_score = 0.0
    parsed_outputs = []
    
    for result in testcase_results:
        parsed_output = parse_testcase_output(result)
        parsed_outputs.append(parsed_output)
        score = calculate_testcase_score(result, parsed_output, baseline_data)
        logger.info(f"Test case {result['testcase']} score: {score:.4f}")
        total_score += score
    
    # 平均评分（加速比的平均值）
    avg_score = total_score / len(testcase_results) if testcase_results else 0
    
    # 生成详细输出
    detail = {
        'timestamp': datetime.now().isoformat(),
        'total_testcases': len(testcase_results),
        'passed_testcases': sum(1 for r in testcase_results if r['exit_code'] == 0),
        'failed_testcases': sum(1 for r in testcase_results if r['exit_code'] != 0),
        'testcase_details': [
            {
                'testcase': r['testcase'],
                'exit_code': r['exit_code'],
                'score': calculate_testcase_score(r, parsed_outputs[i], baseline_data),
                'avg_execution_time': parsed_outputs[i].get('avg_execution_time')
            }
            for i, r in enumerate(testcase_results)
        ]
    }
    
    # 生成最终结果
    final_result = {
        'verdict': verdict,
        'rank': {
            'rank': avg_score  # 使用平均加速比作为排名依据
        },
        'detail': detail
    }
    
    return final_result

# 主函数
def main():
    logger.info("Starting case evaluation...")
    
    # 创建输出目录
    create_output_dir()
    
    # 获取所有测试用例
    testcase_files = get_testcase_files()
    
    if not testcase_files:
        logger.warning("No test cases found in testcases directory")
        final_result = {
            'verdict': 'WA',
            'rank': {'rank': 0.0},
            'detail': {
                'timestamp': datetime.now().isoformat(),
                'total_testcases': 0,
                'passed_testcases': 0,
                'failed_testcases': 0,
                'testcase_details': []
            }
        }
        print(json.dumps(final_result, indent=2))
        return
    
    # 运行所有测试用例
    testcase_results = []
    for testcase_file in testcase_files:
        result = run_testcase(testcase_file)
        testcase_results.append(result)
    
    # 生成最终结果
    final_result = generate_final_result(testcase_results)
    
    # 打印结果（用于外部程序获取）
    print(json.dumps(final_result, indent=2))
    
    logger.info("Case evaluation completed")

if __name__ == '__main__':
    main()
