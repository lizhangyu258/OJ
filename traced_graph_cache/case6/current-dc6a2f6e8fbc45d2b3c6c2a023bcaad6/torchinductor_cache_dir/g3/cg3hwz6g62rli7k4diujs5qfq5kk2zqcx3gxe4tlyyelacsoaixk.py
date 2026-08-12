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


mlir_fused_add_div_mean_mul_pow_sqrt_55 = async_compile.mlir_auto_fallback('mlir_fused_add_div_mean_mul_pow_sqrt_55', '''
module {
  func.func @mlir_fused_add_div_mean_mul_pow_sqrt_55(%arg0: tensor<32x512xf32>, %arg1: tensor<512xf32>) -> tensor<32x512xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 9.9999999999999995E-7 : f64
    %cst_1 = arith.constant 5.120000e+02 : f32
    %0 = tensor.empty() : tensor<32x512xi64>
    %1 = linalg.fill ins(%c2_i64 : i64) outs(%0 : tensor<32x512xi64>) -> tensor<32x512xi64>
    %2 = tensor.empty() : tensor<32x512xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%1 : tensor<32x512xi64>) outs(%2 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %3 : tensor<32x512xf32>, tensor<32x512xf32>) outs(%2 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %5 = tensor.empty() : tensor<32xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<32xf32>) -> tensor<32xf32>
    %reduced = linalg.reduce ins(%4 : tensor<32x512xf32>) outs(%6 : tensor<32xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %20 = arith.addf %in, %init : f32
        linalg.yield %20 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
    %7 = tensor.empty() : tensor<32x1xf32>
    %8 = linalg.fill ins(%cst_1 : f32) outs(%7 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded, %8 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%7 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %10 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%9 : tensor<32x1xf32>) outs(%7 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %11 = arith.truncf %cst_0 : f64 to f32
    %12 = linalg.fill ins(%11 : f32) outs(%7 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %13 = tensor.empty() : tensor<32x1xi64>
    %14 = linalg.fill ins(%c1_i64 : i64) outs(%13 : tensor<32x1xi64>) -> tensor<32x1xi64>
    %15 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%14 : tensor<32x1xi64>) outs(%7 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%12, %15 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%7 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%10, %16 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%7 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %collapsed = tensor.collapse_shape %17 [[0, 1]] : tensor<32x1xf32> into tensor<32xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<32xf32>) outs(%2 : tensor<32x512xf32>) dimensions = [1] 
    %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %broadcasted : tensor<32x512xf32>, tensor<32x512xf32>) outs(%2 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %broadcasted_2 = linalg.broadcast ins(%arg1 : tensor<512xf32>) outs(%2 : tensor<32x512xf32>) dimensions = [0] 
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%18, %broadcasted_2 : tensor<32x512xf32>, tensor<32x512xf32>) outs(%2 : tensor<32x512xf32>) -> tensor<32x512xf32>
    return %19 : tensor<32x512xf32>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': False, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'czfhyxv7osazsrhfyvyexya5f7kl7rmpa2apktrqp2siiv7bo5qp', 'num_call_functions': 2, 'signature': {0: '*f32', 1: '*f32', 2: '*f32'}, 'ranks': [2, 1, 2], 'kernel_hash': 'fca53d56ee6bfc68b1d5af01cf0a46386de13833eafc92eb4f4770ad23657cd5'})
torch_npu.npu.set_device(0)


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    buf1 = empty_strided((32, 512), (512, 1), device='npu', dtype=torch.float32)
    stream0 = get_raw_stream(0)
    mlir_fused_add_div_mean_mul_pow_sqrt_55.run(arg0_1, arg1_1, buf1, stream=stream0)
    del arg0_1
    del arg1_1
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 512), (512, 1), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((512, ), (1, ), device='npu:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
