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


mlir_fused__to_copy_add_mean_mul_pow_rsqrt_sub_39 = async_compile.mlir_auto_fallback('mlir_fused__to_copy_add_mean_mul_pow_rsqrt_sub_39', '''
module {
  func.func @mlir_fused__to_copy_add_mean_mul_pow_rsqrt_sub_39(%arg0: tensor<256x2048xf16>, %arg1: tensor<256x2048xf16>, %arg2: tensor<1x2048xf16>) -> tensor<256x2048xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-02 : f64
    %cst_1 = arith.constant 1.000000e-03 : f64
    %cst_2 = arith.constant 2.500000e-01 : f32
    %cst_3 = arith.constant 2.048000e+03 : f32
    %0 = tensor.empty() : tensor<256x2048xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %2 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %4 = tensor.empty() : tensor<256x2048xi64>
    %5 = linalg.fill ins(%c1_i64 : i64) outs(%4 : tensor<256x2048xi64>) -> tensor<256x2048xi64>
    %6 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%5 : tensor<256x2048xi64>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %6 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %7 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %9 = tensor.empty() : tensor<1x2048xf32>
    %10 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg2 : tensor<1x2048xf16>) outs(%9 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %collapsed = tensor.collapse_shape %10 [[0, 1]] : tensor<1x2048xf32> into tensor<2048xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<2048xf32>) outs(%0 : tensor<256x2048xf32>) dimensions = [0] 
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%8, %broadcasted : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %12 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %12 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%13, %6 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%11, %14 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %16 = tensor.empty() : tensor<256xf32>
    %17 = linalg.fill ins(%cst : f32) outs(%16 : tensor<256xf32>) -> tensor<256xf32>
    %reduced = linalg.reduce ins(%15 : tensor<256x2048xf32>) outs(%17 : tensor<256xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %41 = arith.addf %in, %init : f32
        linalg.yield %41 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [256, 1] : tensor<256xf32> into tensor<256x1xf32>
    %18 = tensor.empty() : tensor<256x1xf32>
    %19 = linalg.fill ins(%cst_3 : f32) outs(%18 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded, %19 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%18 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %collapsed_4 = tensor.collapse_shape %20 [[0, 1]] : tensor<256x1xf32> into tensor<256xf32>
    %broadcasted_5 = linalg.broadcast ins(%collapsed_4 : tensor<256xf32>) outs(%0 : tensor<256x2048xf32>) dimensions = [1] 
    %21 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_5, %6 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%15, %21 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %23 = linalg.fill ins(%c2_i64 : i64) outs(%4 : tensor<256x2048xi64>) -> tensor<256x2048xi64>
    %24 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%23 : tensor<256x2048xi64>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %25 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%22, %24 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %reduced_6 = linalg.reduce ins(%25 : tensor<256x2048xf32>) outs(%17 : tensor<256xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %41 = arith.addf %in, %init : f32
        linalg.yield %41 : f32
      }
    %expanded_7 = tensor.expand_shape %reduced_6 [[0, 1]] output_shape [256, 1] : tensor<256xf32> into tensor<256x1xf32>
    %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded_7, %19 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%18 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %27 = arith.truncf %cst_1 : f64 to f32
    %28 = linalg.fill ins(%27 : f32) outs(%18 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %29 = tensor.empty() : tensor<256x1xi64>
    %30 = linalg.fill ins(%c1_i64 : i64) outs(%29 : tensor<256x1xi64>) -> tensor<256x1xi64>
    %31 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%30 : tensor<256x1xi64>) outs(%18 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %32 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%28, %31 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%18 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %33 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%26, %32 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%18 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %34 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%33 : tensor<256x1xf32>) outs(%18 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %collapsed_8 = tensor.collapse_shape %34 [[0, 1]] : tensor<256x1xf32> into tensor<256xf32>
    %broadcasted_9 = linalg.broadcast ins(%collapsed_8 : tensor<256xf32>) outs(%0 : tensor<256x2048xf32>) dimensions = [1] 
    %35 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%22, %broadcasted_9 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %36 = arith.truncf %cst_0 : f64 to f32
    %37 = linalg.fill ins(%36 : f32) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %38 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%15, %37 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %39 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%38, %6 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %40 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%35, %39 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    return %40 : tensor<256x2048xf32>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': False, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'cbdexsnnn7tk6cp7ffv6eq5o6lti4mzclzfulbpwybflqax3rm6i', 'num_call_functions': 2, 'signature': {0: '*f16', 1: '*f16', 2: '*f16', 3: '*f32'}, 'ranks': [2, 2, 2, 2], 'kernel_hash': 'd85e1c546aeb6c567cd3d04f127ae49ead3bbf598c7e6fcad42c13a7fb197788'})
torch_npu.npu.set_device(0)


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    buf2 = empty_strided((256, 2048), (2048, 1), device='npu', dtype=torch.float32)
    stream0 = get_raw_stream(0)
    mlir_fused__to_copy_add_mean_mul_pow_rsqrt_sub_39.run(arg0_1, arg1_1, arg2_1, buf2, stream=stream0)
    del arg0_1
    del arg1_1
    del arg2_1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((256, 2048), (2048, 1), device='npu:0', dtype=torch.float16)
    arg1_1 = rand_strided((256, 2048), (2048, 1), device='npu:0', dtype=torch.float16)
    arg2_1 = rand_strided((1, 2048), (2048, 1), device='npu:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
