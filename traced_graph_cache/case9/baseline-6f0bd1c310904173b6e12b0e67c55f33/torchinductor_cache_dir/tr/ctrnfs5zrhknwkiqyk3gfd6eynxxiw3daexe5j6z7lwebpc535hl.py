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


mlir_fused__to_copy_add_mean_mul_relu_sigmoid_sub_tanh_61 = async_compile.mlir_auto_fallback('mlir_fused__to_copy_add_mean_mul_relu_sigmoid_sub_tanh_61', '''
module {
  func.func @mlir_fused__to_copy_add_mean_mul_relu_sigmoid_sub_tanh_61(%arg0: tensor<256x2048xf16>, %arg1: tensor<1x2048xf16>, %arg2: tensor<256x2048xf16>) -> tensor<256x2048xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 2.000000e-01 : f64
    %cst_2 = arith.constant 5.000000e-01 : f32
    %cst_3 = arith.constant 2.500000e-01 : f32
    %cst_4 = arith.constant 1.250000e-01 : f32
    %cst_5 = arith.constant 2.048000e+03 : f32
    %0 = tensor.empty() : tensor<256x2048xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %2 = tensor.empty() : tensor<1x2048xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<1x2048xf16>) outs(%2 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %collapsed = tensor.collapse_shape %3 [[0, 1]] : tensor<1x2048xf32> into tensor<2048xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<2048xf32>) outs(%0 : tensor<256x2048xf32>) dimensions = [0] 
    %4 = tensor.empty() : tensor<256x2048xi64>
    %5 = linalg.fill ins(%c1_i64 : i64) outs(%4 : tensor<256x2048xi64>) -> tensor<256x2048xi64>
    %6 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%5 : tensor<256x2048xi64>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %6 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %7 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %9 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%8 : tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %10 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%9 : tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%10, %cst : tensor<256x2048xf32>, f32) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %11 : f32, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %13 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg2 : tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%13, %7 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %15 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tanh>} ins(%14 : tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%12, %15 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%13, %6 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %17 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%18, %7 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %20 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} ins(%19 : tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %21 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%20, %21 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%22, %6 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %24 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%16, %23 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %25 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%12, %20 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %26 = linalg.fill ins(%cst_3 : f32) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%25, %26 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%27, %6 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %29 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%24, %28 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %30 = linalg.fill ins(%cst_4 : f32) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %31 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %30 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %32 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%31, %6 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %33 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%29, %32 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %34 = tensor.empty() : tensor<256xf32>
    %35 = linalg.fill ins(%cst_0 : f32) outs(%34 : tensor<256xf32>) -> tensor<256xf32>
    %reduced = linalg.reduce ins(%33 : tensor<256x2048xf32>) outs(%35 : tensor<256xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %44 = arith.addf %in, %init : f32
        linalg.yield %44 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [256, 1] : tensor<256xf32> into tensor<256x1xf32>
    %36 = tensor.empty() : tensor<256x1xf32>
    %37 = linalg.fill ins(%cst_5 : f32) outs(%36 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %38 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded, %37 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%36 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %39 = arith.truncf %cst_1 : f64 to f32
    %40 = linalg.fill ins(%39 : f32) outs(%36 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %41 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%38, %40 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%36 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %collapsed_6 = tensor.collapse_shape %41 [[0, 1]] : tensor<256x1xf32> into tensor<256xf32>
    %broadcasted_7 = linalg.broadcast ins(%collapsed_6 : tensor<256xf32>) outs(%0 : tensor<256x2048xf32>) dimensions = [1] 
    %42 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_7, %6 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %43 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%33, %42 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    return %43 : tensor<256x2048xf32>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': False, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'cumqaps3wo36o72zcirzpmjagvisq6n2i7kkhuqwhpsn3wgfrihp', 'num_call_functions': 2, 'signature': {0: '*f16', 1: '*f16', 2: '*f16', 3: '*f32'}, 'ranks': [2, 2, 2, 2], 'kernel_hash': '9ba493064c8d4cdc1f2ec09454aea34e77b57e0c145216baea3f49483a684b4b'})
torch_npu.npu.set_device(0)


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    buf1 = empty_strided((256, 2048), (2048, 1), device='npu', dtype=torch.float32)
    stream0 = get_raw_stream(0)
    mlir_fused__to_copy_add_mean_mul_relu_sigmoid_sub_tanh_61.run(arg0_1, arg2_1, arg1_1, buf1, stream=stream0)
    del arg0_1
    del arg1_1
    del arg2_1
    return (buf1, )


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
