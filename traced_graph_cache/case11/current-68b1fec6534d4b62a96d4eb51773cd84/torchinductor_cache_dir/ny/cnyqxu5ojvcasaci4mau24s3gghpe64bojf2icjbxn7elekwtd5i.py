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


mlir_fused__to_copy_add_mean_mul_relu_sub_18 = async_compile.mlir_auto_fallback('mlir_fused__to_copy_add_mean_mul_relu_sub_18', '''
module {
  func.func @mlir_fused__to_copy_add_mean_mul_relu_sub_18(%arg0: tensor<?x?xf16>, %arg1: tensor<1x?xf16>, %arg2: i64, %arg3: i64, %arg4: tensor<?x1xf16>) -> tensor<?x?xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %c1_i64 = arith.constant 1 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.250000e-01 : f32
    %cst_1 = arith.constant 7.500000e-01 : f32
    %cst_2 = arith.constant 2.500000e-01 : f32
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %dim_3 = tensor.dim %arg0, %c1 : tensor<?x?xf16>
    %0 = tensor.empty(%dim, %dim_3) : tensor<?x?xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<?x?xf16>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %dim_4 = tensor.dim %arg1, %c1 : tensor<1x?xf16>
    %2 = tensor.empty(%dim_4) : tensor<1x?xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<1x?xf16>) outs(%2 : tensor<1x?xf32>) -> tensor<1x?xf32>
    %4 = arith.index_cast %arg2 : i64 to index
    %5 = tensor.empty(%4, %dim_4) : tensor<?x?xf32>
    %collapsed = tensor.collapse_shape %3 [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<?xf32>) outs(%5 : tensor<?x?xf32>) dimensions = [0] 
    %6 = tensor.empty(%dim, %dim_3) : tensor<?x?xi64>
    %7 = linalg.fill ins(%c1_i64 : i64) outs(%6 : tensor<?x?xi64>) -> tensor<?x?xi64>
    %8 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%7 : tensor<?x?xi64>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %9 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %dim_5 = tensor.dim %arg4, %c0 : tensor<?x1xf16>
    %11 = tensor.empty(%dim_5) : tensor<?x1xf32>
    %12 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg4 : tensor<?x1xf16>) outs(%11 : tensor<?x1xf32>) -> tensor<?x1xf32>
    %13 = arith.index_cast %arg3 : i64 to index
    %14 = tensor.empty(%dim_5, %13) : tensor<?x?xf32>
    %collapsed_6 = tensor.collapse_shape %12 [[0, 1]] : tensor<?x1xf32> into tensor<?xf32>
    %broadcasted_7 = linalg.broadcast ins(%collapsed_6 : tensor<?xf32>) outs(%14 : tensor<?x?xf32>) dimensions = [1] 
    %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%10, %broadcasted_7 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %16 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} ins(%15 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %17 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %17 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%18, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%16, %19 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %21 = tensor.empty(%dim) : tensor<?xf32>
    %22 = linalg.fill ins(%cst : f32) outs(%21 : tensor<?xf32>) -> tensor<?xf32>
    %reduced = linalg.reduce ins(%20 : tensor<?x?xf32>) outs(%22 : tensor<?xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %37 = arith.addf %in, %init : f32
        linalg.yield %37 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [%dim, 1] : tensor<?xf32> into tensor<?x1xf32>
    %23 = tensor.empty(%dim) : tensor<?x1xi64>
    %24 = linalg.fill ins(%arg3 : i64) outs(%23 : tensor<?x1xi64>) -> tensor<?x1xi64>
    %25 = tensor.empty(%dim) : tensor<?x1xf32>
    %26 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%24 : tensor<?x1xi64>) outs(%25 : tensor<?x1xf32>) -> tensor<?x1xf32>
    %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded, %26 : tensor<?x1xf32>, tensor<?x1xf32>) outs(%25 : tensor<?x1xf32>) -> tensor<?x1xf32>
    %28 = tensor.empty(%dim, %13) : tensor<?x?xf32>
    %collapsed_8 = tensor.collapse_shape %27 [[0, 1]] : tensor<?x1xf32> into tensor<?xf32>
    %broadcasted_9 = linalg.broadcast ins(%collapsed_8 : tensor<?xf32>) outs(%28 : tensor<?x?xf32>) dimensions = [1] 
    %29 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_9, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %30 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%20, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %31 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %32 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%30, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %33 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %34 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%20, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %35 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%34, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %36 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%32, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %36 : tensor<?x?xf32>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': True, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'cipv47d4ouavr7hbmrjynimgsle2ph2i6fnrpwv7x5cxsvvxksvw', 'num_call_functions': 2, 'signature': {0: '*f16', 1: '*f16', 2: '*!torch.int', 3: '*!torch.int', 4: '*f16', 5: '*f32'}, 'ranks': [2, 2, 1, 1, 2, 2], 'kernel_hash': '4da41aa23dfb58c097badab647279d9dff206eb5e413c67c41a822d044e9e7d1'})
torch_npu.npu.set_device(0)


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    buf1 = empty_strided((s0, s1), (s1, 1), device='npu', dtype=torch.float32)
    stream0 = get_raw_stream(0)
    mlir_fused__to_copy_add_mean_mul_relu_sub_18.run(arg2_1, arg3_1, s0, s1, arg4_1, buf1, stream=stream0)
    del arg2_1
    del arg3_1
    del arg4_1
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 192
    arg1_1 = 1024
    arg2_1 = rand_strided((192, 1024), (1024, 1), device='npu:0', dtype=torch.float16)
    arg3_1 = rand_strided((1, 1024), (1024, 1), device='npu:0', dtype=torch.float16)
    arg4_1 = rand_strided((192, 1), (1, 1), device='npu:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
