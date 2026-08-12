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


mlir_fused_add_mean_mul_pow_rsqrt_sub_58 = async_compile.mlir_auto_fallback('mlir_fused_add_mean_mul_pow_rsqrt_sub_58', '''
module {
  func.func @mlir_fused_add_mean_mul_pow_rsqrt_sub_58(%arg0: tensor<256x2048xf32>) -> tensor<256x2048xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-02 : f64
    %cst_1 = arith.constant 1.000000e-04 : f64
    %cst_2 = arith.constant 2.048000e+03 : f32
    %cst_3 = arith.constant 7.500000e-01 : f32
    %cst_4 = arith.constant 1.250000e-01 : f32
    %0 = tensor.empty() : tensor<256xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256xf32>) -> tensor<256xf32>
    %reduced = linalg.reduce ins(%arg0 : tensor<256x2048xf32>) outs(%1 : tensor<256xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %36 = arith.addf %in, %init : f32
        linalg.yield %36 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [256, 1] : tensor<256xf32> into tensor<256x1xf32>
    %2 = tensor.empty() : tensor<256x1xf32>
    %3 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded, %3 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%2 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %5 = tensor.empty() : tensor<256x2048xf32>
    %collapsed = tensor.collapse_shape %4 [[0, 1]] : tensor<256x1xf32> into tensor<256xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<256xf32>) outs(%5 : tensor<256x2048xf32>) dimensions = [1] 
    %6 = tensor.empty() : tensor<256x2048xi64>
    %7 = linalg.fill ins(%c1_i64 : i64) outs(%6 : tensor<256x2048xi64>) -> tensor<256x2048xi64>
    %8 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%7 : tensor<256x2048xi64>) outs(%5 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %8 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%5 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %9 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%5 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %11 = linalg.fill ins(%c2_i64 : i64) outs(%6 : tensor<256x2048xi64>) -> tensor<256x2048xi64>
    %12 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%11 : tensor<256x2048xi64>) outs(%5 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %13 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%10, %12 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%5 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %reduced_5 = linalg.reduce ins(%13 : tensor<256x2048xf32>) outs(%1 : tensor<256xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %36 = arith.addf %in, %init : f32
        linalg.yield %36 : f32
      }
    %expanded_6 = tensor.expand_shape %reduced_5 [[0, 1]] output_shape [256, 1] : tensor<256xf32> into tensor<256x1xf32>
    %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded_6, %3 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%2 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %15 = arith.truncf %cst_1 : f64 to f32
    %16 = linalg.fill ins(%15 : f32) outs(%2 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %17 = tensor.empty() : tensor<256x1xi64>
    %18 = linalg.fill ins(%c1_i64 : i64) outs(%17 : tensor<256x1xi64>) -> tensor<256x1xi64>
    %19 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%18 : tensor<256x1xi64>) outs(%2 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%16, %19 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%2 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %21 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%14, %20 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%2 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %22 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%21 : tensor<256x1xf32>) outs(%2 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %collapsed_7 = tensor.collapse_shape %22 [[0, 1]] : tensor<256x1xf32> into tensor<256xf32>
    %broadcasted_8 = linalg.broadcast ins(%collapsed_7 : tensor<256xf32>) outs(%5 : tensor<256x2048xf32>) dimensions = [1] 
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%10, %broadcasted_8 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%5 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %24 = linalg.fill ins(%cst_3 : f32) outs(%5 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %25 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%23, %24 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%5 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %26 = linalg.fill ins(%cst_4 : f32) outs(%5 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %26 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%5 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%27, %8 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%5 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %29 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%25, %28 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%5 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %reduced_9 = linalg.reduce ins(%29 : tensor<256x2048xf32>) outs(%1 : tensor<256xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %36 = arith.addf %in, %init : f32
        linalg.yield %36 : f32
      }
    %expanded_10 = tensor.expand_shape %reduced_9 [[0, 1]] output_shape [256, 1] : tensor<256xf32> into tensor<256x1xf32>
    %30 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded_10, %3 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%2 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %31 = arith.truncf %cst_0 : f64 to f32
    %32 = linalg.fill ins(%31 : f32) outs(%2 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %33 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%30, %32 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%2 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %collapsed_11 = tensor.collapse_shape %33 [[0, 1]] : tensor<256x1xf32> into tensor<256xf32>
    %broadcasted_12 = linalg.broadcast ins(%collapsed_11 : tensor<256xf32>) outs(%5 : tensor<256x2048xf32>) dimensions = [1] 
    %34 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_12, %8 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%5 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %35 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%29, %34 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%5 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    return %35 : tensor<256x2048xf32>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': False, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'c45t5fhwpbzhyeqhhn4ab3omwsmbtv5kgx3jbpdpoaz7c2u62zoi', 'num_call_functions': 2, 'signature': {0: '*f32', 1: '*f32'}, 'ranks': [2, 2], 'kernel_hash': '4f8e7ef9050a59e7cb9b7916a69e672f80402229f2eb0f5d43367ef3767e9cea'})
torch_npu.npu.set_device(0)


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    buf3 = empty_strided((256, 2048), (2048, 1), device='npu', dtype=torch.float32)
    stream0 = get_raw_stream(0)
    mlir_fused_add_mean_mul_pow_rsqrt_sub_58.run(arg0_1, buf3, stream=stream0)
    del arg0_1
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((256, 2048), (2048, 1), device='npu:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
