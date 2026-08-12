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


mlir_fused__to_copy_add_mean_mul_relu_49 = async_compile.mlir_auto_fallback('mlir_fused__to_copy_add_mean_mul_relu_49', '''
module {
  func.func @mlir_fused__to_copy_add_mean_mul_relu_49(%arg0: tensor<256x2048xf16>, %arg1: tensor<1x2048xf16>, %arg2: tensor<256x2048xf16>, %arg3: tensor<256x1xf16>) -> tensor<256x2048xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 2.500000e-01 : f32
    %cst_2 = arith.constant 2.048000e+03 : f32
    %cst_3 = arith.constant 1.250000e-01 : f32
    %0 = tensor.empty() : tensor<256x2048xf16>
    %collapsed = tensor.collapse_shape %arg1 [[0, 1]] : tensor<1x2048xf16> into tensor<2048xf16>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<2048xf16>) outs(%0 : tensor<256x2048xf16>) dimensions = [0] 
    %1 = tensor.empty() : tensor<256x2048xi64>
    %2 = linalg.fill ins(%c1_i64 : i64) outs(%1 : tensor<256x2048xi64>) -> tensor<256x2048xi64>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%2 : tensor<256x2048xi64>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %3 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %4 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%5, %arg2 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %collapsed_4 = tensor.collapse_shape %arg3 [[0, 1]] : tensor<256x1xf16> into tensor<256xf16>
    %broadcasted_5 = linalg.broadcast ins(%collapsed_4 : tensor<256xf16>) outs(%0 : tensor<256x2048xf16>) dimensions = [1] 
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_5, %3 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%6, %7 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %9 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} ins(%8 : tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %10 = tensor.empty() : tensor<256x2048xf32>
    %11 = linalg.fill ins(%cst_1 : f32) outs(%10 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %12 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%11 : tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%8, %12 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%13, %3 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%9, %14 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %16 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%15 : tensor<256x2048xf16>) outs(%10 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %17 = tensor.empty() : tensor<256xf32>
    %18 = linalg.fill ins(%cst_0 : f32) outs(%17 : tensor<256xf32>) -> tensor<256xf32>
    %reduced = linalg.reduce ins(%16 : tensor<256x2048xf32>) outs(%18 : tensor<256xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %30 = arith.addf %in, %init : f32
        linalg.yield %30 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [256, 1] : tensor<256xf32> into tensor<256x1xf32>
    %19 = tensor.empty() : tensor<256x1xf32>
    %20 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %21 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded, %20 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%19 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %22 = tensor.empty() : tensor<256x1xf16>
    %23 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%21 : tensor<256x1xf32>) outs(%22 : tensor<256x1xf16>) -> tensor<256x1xf16>
    %24 = linalg.fill ins(%cst_3 : f32) outs(%19 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %25 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%24 : tensor<256x1xf32>) outs(%22 : tensor<256x1xf16>) -> tensor<256x1xf16>
    %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%23, %25 : tensor<256x1xf16>, tensor<256x1xf16>) outs(%22 : tensor<256x1xf16>) -> tensor<256x1xf16>
    %collapsed_6 = tensor.collapse_shape %26 [[0, 1]] : tensor<256x1xf16> into tensor<256xf16>
    %broadcasted_7 = linalg.broadcast ins(%collapsed_6 : tensor<256xf16>) outs(%0 : tensor<256x2048xf16>) dimensions = [1] 
    %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_7, %3 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%15, %27 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %29 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%28 : tensor<256x2048xf16>) outs(%10 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    return %29 : tensor<256x2048xf32>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': False, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'cfqtjcm72l4jedlakdhwdxba63ohmgfpnfdzyda6thlcdfhsp345', 'num_call_functions': 2, 'signature': {0: '*f16', 1: '*f16', 2: '*f16', 3: '*f16', 4: '*f32'}, 'ranks': [2, 2, 2, 2, 2], 'kernel_hash': '799ce85ce619ef0e79a578eea4d3cd759dc70bb30a7182b276a48b5574d18c5c'})
torch_npu.npu.set_device(0)


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    buf1 = empty_strided((256, 2048), (2048, 1), device='npu', dtype=torch.float32)
    stream0 = get_raw_stream(0)
    mlir_fused__to_copy_add_mean_mul_relu_49.run(arg0_1, arg1_1, arg2_1, arg3_1, buf1, stream=stream0)
    del arg0_1
    del arg1_1
    del arg2_1
    del arg3_1
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((256, 2048), (2048, 1), device='npu:0', dtype=torch.float16)
    arg1_1 = rand_strided((1, 2048), (2048, 1), device='npu:0', dtype=torch.float16)
    arg2_1 = rand_strided((256, 2048), (2048, 1), device='npu:0', dtype=torch.float16)
    arg3_1 = rand_strided((256, 1), (1, 1), device='npu:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
