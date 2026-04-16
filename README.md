# bishengir-compile 性能评测项目

## 项目简介

该项目用于评测bishengir-compile编译器对接torch_npu生成的子图，进行自动编译优化后，对算子端到端的性能提升效果。通过运行一系列测试用例，收集编译和执行性能数据，并生成统一的评测报告。

## 项目结构

```
OJ/
├── case_judge.py        # 主评测脚本
├── run.sh               # 运行脚本
├── testcases/           # 测试用例目录
│   └── case1.py         # 示例测试用例
├── outputs/             # 输出目录（自动生成）
└── .gitignore           # Git忽略文件
```

## 环境要求

- Python 3.6+
- PyYAML
- PyTorch
- torch_npu
- bishengir-compile 和 bishengir-opt 工具

## 使用方式

### 1. 准备工作

确保bishengir-compile和bishengir-opt工具已放置在项目根目录下。

安装 Python 依赖：

```bash
python3 -m pip install -r requirements.txt
```

### 2. 运行评测

有两种方式运行评测：

#### 方式一：直接运行主脚本

```bash
python3 ./case_judge.py
```

#### 方式二：使用run.sh脚本（推荐）

```bash
chmod +x run.sh
./run.sh
```

run.sh脚本会自动：
- 检查bishengir-compile和bishengir-opt工具是否存在
- 添加执行权限
- 将工具路径添加到环境变量
- 运行主评测脚本

### 3. 查看评测结果

评测完成后，会在控制台输出JSON格式的评测结果，同时在`outputs/`目录下保存每个测试用例的详细输出。

## 测试用例编写

在`testcases/`目录下创建Python脚本作为测试用例，脚本应：

1. 设置环境变量（如需要）
2. 导入必要的库（torch, torch_npu等）
3. 定义测试模型或算子
4. 运行原始计算和编译优化后的计算
5. 输出性能指标或相关信息

示例测试用例（testcases/case1.py）：

```python
import os
# 环境变量调用需要在torch_npu初始化之前
os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'mlir'

import torch
import torch_npu
from torch._inductor.utils import run_and_get_code
import torch_npu._inductor
import torch
import triton

# config导入在compile执行之前
torch._inductor.config.npu_backend = "mlir"

# 定义模型
def op_calc(x, y):
    return x * y
x = torch.randn((3,), requires_grad=False, dtype=torch.float32, device="npu")
y = torch.randn((3,), requires_grad=False, dtype=torch.float32, device="npu")
std_out = op_calc(x, y)

# options调用，修改compile参数
compile_func = torch.compile(op_calc, options={"npu_backend": "mlir"})
compile_out, codes = run_and_get_code(compile_func, x, y)
print(codes[0])
```

## 评测结果说明

评测结果以JSON格式输出，包含以下字段：

```json
{
  "verdict": "AC" or "WA",
  "rank": {
    "rank": final_marks
  },
  "detail": {
    "timestamp": "2026-03-26T12:31:33.107036",
    "total_testcases": 1,
    "passed_testcases": 0,
    "failed_testcases": 1,
    "testcase_details": [
      {
        "testcase": "case1.py",
        "exit_code": 1,
        "score": 0.0,
        "has_output": false,
        "has_error": true
      }
    ]
  }
}
```

- `verdict`：评测结果，"AC"表示所有测试用例通过，"WA"表示存在失败的测试用例
- `rank.rank`：最终评分，目前为测试用例的平均得分
- `detail`：详细评测信息，包含：
  - `timestamp`：评测时间
  - `total_testcases`：测试用例总数
  - `passed_testcases`：通过的测试用例数
  - `failed_testcases`：失败的测试用例数
  - `testcase_details`：每个测试用例的详细结果

## 扩展建议

### 性能指标扩展

当前脚本提供了以下预留接口，用于扩展性能指标的提取和评分：

1. `parse_testcase_output()`：从测试输出中提取性能指标（如编译时间、执行时间、加速比等）
2. `calculate_testcase_score()`：根据性能指标加权计算评分

### 功能扩展

- 添加性能对比功能，对比优化前后的性能差异
- 实现更复杂的评分算法，考虑多种性能指标
- 添加结果可视化功能，生成性能对比图表
- 支持并行运行测试用例，提高评测效率

## 运行测试

可以使用 `unittest` 运行核心逻辑测试：

```bash
python3 -m unittest discover -s tests
```

## 注意事项

- 确保测试用例脚本具有正确的执行权限
- 评测过程中会生成outputs目录，该目录已添加到.gitignore中
- 每个测试用例的输出会保存到outputs目录下，格式为`测试用例名.out`（标准输出）和`测试用例名.err`（错误输出）
- 测试用例运行超时时间为5分钟
