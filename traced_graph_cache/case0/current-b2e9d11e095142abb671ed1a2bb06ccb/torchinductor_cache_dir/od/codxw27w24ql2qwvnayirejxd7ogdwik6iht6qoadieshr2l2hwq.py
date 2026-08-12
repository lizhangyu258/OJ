from torch_npu._C import _npu_getCurrentRawStream as get_raw_stream


from ctypes import c_void_p, c_long
import torch
import torch_npu
import math
import random
import os
os.environ["TORCHINDUCTOR_NPU_BACKEND"] = 'mlir'
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.codecache import CustomAsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
from torch.utils._sympy.functions import FloatTrueDiv
from torch.utils._sympy.functions import IntTrueDiv

has_initialized = False
aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
async_compile = CustomAsyncCompile()


mlir_fused_add_mean_mul_sub_sum_0 = async_compile.mlir_auto_fallback('mlir_fused_add_mean_mul_sub_sum_0', '''
module {
  func.func @mlir_fused_add_mean_mul_sub_sum_0(%arg0: tensor<16x1000xf32>, %arg1: tensor<16x1000xf32>) -> (tensor<16xf32>, tensor<16xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e-01 : f64
    %cst_1 = arith.constant 5.000000e-01 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant 1.500000e+00 : f32
    %0 = tensor.empty() : tensor<16x1000xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %2 = tensor.empty() : tensor<16x1000xi64>
    %3 = linalg.fill ins(%c1_i64 : i64) outs(%2 : tensor<16x1000xi64>) -> tensor<16x1000xi64>
    %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%3 : tensor<16x1000xi64>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %4 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %5 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %4 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %7 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg0 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %10 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %10 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%11, %4 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%9, %12 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %14 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%14, %4 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%13, %15 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%8, %16 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%16, %4 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%8, %18 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %20 = arith.truncf %cst_0 : f64 to f32
    %21 = linalg.fill ins(%20 : f32) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%19, %21 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%22, %4 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %24 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%17, %23 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %25 = linalg.fill ins(%cst_3 : f32) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %25 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%26, %4 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%24, %27 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %29 = tensor.empty() : tensor<16xf32>
    %30 = linalg.fill ins(%cst : f32) outs(%29 : tensor<16xf32>) -> tensor<16xf32>
    %reduced = linalg.reduce ins(%28 : tensor<16x1000xf32>) outs(%30 : tensor<16xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %31 = arith.addf %in, %init : f32
        linalg.yield %31 : f32
      }
    return %reduced, %reduced : tensor<16xf32>, tensor<16xf32>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 2, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': False, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'cov2cj3ucdkzuen5zsjdvpis3psgupcqrykmxzr3e2bxupazlieg', 'num_call_functions': 2, 'signature': {0: '*f32', 1: '*f32', 2: '*f32', 3: '*f32'}, 'ranks': [2, 2, 1, 1], 'kernel_hash': '9db924ced36384f605555f5c5d1e7dda825c60c2062643912bd48df233372fe4'})
torch_npu.npu.set_device(0)


mlir_fused_add_mean_mul_sub_1 = async_compile.mlir_auto_fallback('mlir_fused_add_mean_mul_sub_1', '''
module {
  func.func @mlir_fused_add_mean_mul_sub_1(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<f32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+03 : f32
    %cst_1 = arith.constant 1.600000e+01 : f32
    %0 = tensor.empty() : tensor<16xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg1, %1 : tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %3 = tensor.empty() : tensor<16xi64>
    %4 = linalg.fill ins(%c1_i64 : i64) outs(%3 : tensor<16xi64>) -> tensor<16xi64>
    %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%4 : tensor<16xi64>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %5 : tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %6 : tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %8 = tensor.empty() : tensor<f32>
    %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<f32>) -> tensor<f32>
    %reduced = linalg.reduce ins(%7 : tensor<16xf32>) outs(%9 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %12 = arith.addf %in, %init : f32
        linalg.yield %12 : f32
      }
    %10 = linalg.fill ins(%cst_1 : f32) outs(%8 : tensor<f32>) -> tensor<f32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%reduced, %10 : tensor<f32>, tensor<f32>) outs(%8 : tensor<f32>) -> tensor<f32>
    return %11 : tensor<f32>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': False, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'cbi7dsmosxf7ril36j42ps7aty24nka3cwsvjvqdaycpme6whch4', 'num_call_functions': 2, 'signature': {0: '*f32', 1: '*f32', 2: '*f32'}, 'ranks': [1, 1, 0], 'kernel_hash': '9aec5b46501decc1d59842f9ac93f6a00dcb7e3baad7963245d104d2d28ec059'})


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    buf0 = empty_strided((16, ), (1, ), device='npu', dtype=torch.float32)
    buf1 = empty_strided((16, ), (1, ), device='npu', dtype=torch.float32)
    stream0 = get_raw_stream(0)
    mlir_fused_add_mean_mul_sub_sum_0.run(arg0_1, arg1_1, buf0, buf1, stream=stream0)
    del arg0_1
    del arg1_1
    buf3 = empty_strided((), (), device='npu', dtype=torch.float32)
    mlir_fused_add_mean_mul_sub_1.run(buf0, buf1, buf3, stream=stream0)
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 1000), (1000, 1), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, 1000), (1000, 1), device='npu:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
