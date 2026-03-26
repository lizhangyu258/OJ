import os
import subprocess
import json
import glob
import logging
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

# 解析测试用例输出（预留接口，用于后续扩展）
def parse_testcase_output(testcase_result):
    """解析测试用例输出，提取性能指标等信息"""
    # 当前实现为简单示例，后续可根据实际需求扩展
    parsed_data = {
        'has_output': len(testcase_result['stdout']) > 0,
        'has_error': len(testcase_result['stderr']) > 0,
        'output_length': len(testcase_result['stdout']),
        'error_length': len(testcase_result['stderr'])
    }
    
    # 预留：从输出中提取性能指标的逻辑
    # 例如：提取编译时间、执行时间、加速比等
    
    return parsed_data

# 计算测试用例评分（预留接口，用于后续扩展）
def calculate_testcase_score(testcase_result, parsed_output):
    """根据测试用例结果和解析的输出计算评分"""
    # 当前实现为简单示例，后续可根据实际需求扩展
    score = 0.0
    
    # 基础分：测试用例成功运行
    if testcase_result['exit_code'] == 0:
        score += 100.0
    
    # 预留：根据性能指标加权计算评分的逻辑
    # 例如：根据加速比、编译时间等指标进行加权
    
    return score

# 生成最终结果
def generate_final_result(testcase_results):
    """根据所有测试用例的结果生成最终评测结果"""
    # 检查所有测试用例是否成功
    all_passed = all(result['exit_code'] == 0 for result in testcase_results)
    verdict = 'AC' if all_passed else 'WA'
    
    # 计算总评分
    total_score = 0.0
    parsed_outputs = []
    
    for result in testcase_results:
        parsed_output = parse_testcase_output(result)
        parsed_outputs.append(parsed_output)
        score = calculate_testcase_score(result, parsed_output)
        total_score += score
    
    # 平均评分（可根据实际需求调整）
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
                'score': calculate_testcase_score(r, parsed_outputs[i]),
                'has_output': parsed_outputs[i]['has_output'],
                'has_error': parsed_outputs[i]['has_error']
            }
            for i, r in enumerate(testcase_results)
        ]
    }
    
    # 生成最终结果
    final_result = {
        'verdict': verdict,
        'rank': {
            'rank': avg_score  # 暂用平均评分作为排名依据
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
