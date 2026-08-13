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

在项目根目录准备 `config.yaml`，用于指定两套 `bishengir-compile` 和 `bishengir-opt` 所在目录：

```yaml
bin:
  baseline: /usr/local/Ascend/latest/compiler/bin
  current: /coursegrader/submit
```

- `bin.baseline`：基线工具目录，用于计算 `compile_time`
- `bin.current`：当前被测工具目录，用于计算 `current_time`
- `eager_time` 不依赖这两个目录

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

如需保存 OJ 实际调用 `bishengir-compile` 时各个 pass 后的 IR，可运行：

```bash
./run.sh --save-compile-ir
```

该选项不会修改 testcase 或输入 MLIR。它会为实际编译命令追加
`--mlir-print-ir-after-all`，并分别保存 baseline/current 的输出：

```text
traced_graph_cache/<case>/<phase-uuid>/bishengir_cache_dir/
├── <id>-<kernel>.command.json
└── <id>-<kernel>.ir.log
```

`command.json` 记录工作目录和完整编译参数，`ir.log` 保存编译器 stderr 中的
pass-manager IR dump。该模式会增加编译耗时和磁盘占用，只应用于编译分析，
正式性能测量时不要开启。

单次编译默认最多缓存 512 MiB；可通过环境变量调整，例如：

```bash
OJ_BISHENGIR_IR_MAX_BYTES=$((1024 * 1024 * 1024)) ./run.sh --save-compile-ir
```

如需在评测结束后自动删除项目根目录生成的 `prof/` 和 `traced_graph_cache/` 目录：

```bash
./run.sh --clean-up
```

run.sh脚本会自动：
- 调用 `check_bin_path.py` 检查 `config.yaml` 中 `bin.current` 目录下的 `bishengir-compile` 和 `bishengir-opt` 是否存在
- 将当前被测工具目录添加到环境变量
- 运行主评测脚本
- 当传入 `--clean-up` 时，在评测完成后删除项目根目录下的 `prof/` 和 `traced_graph_cache/` 目录

### 3. 查看评测结果

评测完成后，会在控制台输出JSON格式的评测结果。

## 测试用例编写

在`testcases/`目录下创建Python脚本作为测试用例，脚本应：

1. 设置环境变量（如需要）
2. 导入必要的库（torch, torch_npu等）
3. 定义测试模型或算子
4. 提供 `build_testcase()` 函数，返回测试配置字典
5. 由 `case_judge.py` 统一调用 `benchmark()`

示例测试用例（testcases/case1.py）：

```python
import torch

def op_calc(x, y):
    return x * y


def build_testcase():
    device = "npu"
    x = torch.randn((3,), requires_grad=False, dtype=torch.float32, device=device)
    y = torch.randn((3,), requires_grad=False, dtype=torch.float32, device=device)
    return {
        "model_or_func": op_calc,
        "inputs": (x, y),
        "device": device,
    }
```

## 评测结果说明

评测结果以JSON格式输出，包含以下字段：

```json
{
  "verdict": "AC" or "WA",
  "rank": {
    "rank": final_marks
  },
  "detail": "{\"timestamp\":\"2026-03-26T12:31:33.107036\",\"total_testcases\":1,\"passed_testcases\":0,\"failed_testcases\":1}"
}
```

- `verdict`：评测结果，"AC"表示所有测试用例通过，"WA"表示存在失败的测试用例
- `rank.rank`：最终评分，目前为测试用例的平均得分
- `detail`：字符串类型。内容是序列化后的 JSON，可反序列化得到详细评测信息，包含：
  - `timestamp`：评测时间
  - `total_testcases`：测试用例总数
  - `passed_testcases`：通过的测试用例数
  - `failed_testcases`：失败的测试用例数
  - `testcase_details`：每个测试用例的详细结果

## 扩展建议

### 性能指标扩展

当前脚本提供了以下预留接口，用于扩展性能指标的提取和评分：

1. `extract_testcase_metrics()`：从 `benchmark()` 返回的原始结果中提取性能指标
2. `calculate_testcase_score()`：根据性能指标加权计算评分

### 功能扩展

- 添加性能对比功能，对比优化前后的性能差异
- 实现更复杂的评分算法，考虑多种性能指标
- 添加结果可视化功能，生成性能对比图表
- 支持并行运行测试用例，提高评测效率

## 运行测试

可以使用 `unittest` 运行核心逻辑测试：

```bash
python3 -m unittest discover -s unittests
```

## 注意事项

- 确保测试用例脚本提供 `build_testcase()` 函数，并返回包含 `model_or_func` 和 `inputs` 的配置字典
- `case_judge.py` 会统一补充 `artifact_subdir`、默认 `device`、默认 warmup/exec 步数，并调用 `benchmark()`
- `benchmark()` 会使用 `config.yaml` 中的 `bin.baseline` 编译基线程序，使用 `bin.current` 编译当前被测程序
