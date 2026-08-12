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


mlir_fused__to_copy_add_mean_mul_relu_sub_4 = async_compile.mlir_auto_fallback('mlir_fused__to_copy_add_mean_mul_relu_sub_4', '''
module {
  func.func @mlir_fused__to_copy_add_mean_mul_relu_sub_4(%arg0: tensor<256x2048xf16>, %arg1: tensor<256x2048xf16>, %arg2: tensor<1x2048xf16>, %arg3: tensor<256x1xf16>) -> tensor<256x2048xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e-01 : f64
    %cst_2 = arith.constant 9.000000e-01 : f64
    %cst_3 = arith.constant 1.250000e-01 : f32
    %cst_4 = arith.constant 6.250000e-02 : f32
    %cst_5 = arith.constant 2.048000e+03 : f32
    %0 = tensor.empty() : tensor<256x2048xf16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %2 = tensor.empty() : tensor<256x2048xi64>
    %3 = linalg.fill ins(%c1_i64 : i64) outs(%2 : tensor<256x2048xi64>) -> tensor<256x2048xi64>
    %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%3 : tensor<256x2048xi64>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %4 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %5 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %4 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %7 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %collapsed = tensor.collapse_shape %arg2 [[0, 1]] : tensor<1x2048xf16> into tensor<2048xf16>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<2048xf16>) outs(%0 : tensor<256x2048xf16>) dimensions = [0] 
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %4 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%8, %9 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %collapsed_6 = tensor.collapse_shape %arg3 [[0, 1]] : tensor<256x1xf16> into tensor<256xf16>
    %broadcasted_7 = linalg.broadcast ins(%collapsed_6 : tensor<256xf16>) outs(%0 : tensor<256x2048xf16>) dimensions = [1] 
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%10, %broadcasted_7 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %9 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %13 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} ins(%12 : tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %14 = tensor.empty() : tensor<256x2048xf32>
    %15 = linalg.fill ins(%cst_3 : f32) outs(%14 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %16 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%15 : tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%13, %16 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%17, %4 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%11, %18 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %20 = linalg.fill ins(%cst_4 : f32) outs(%14 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %21 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%20 : tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %21 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%22, %4 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %24 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%19, %23 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %25 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%24 : tensor<256x2048xf16>) outs(%14 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %26 = tensor.empty() : tensor<256xf32>
    %27 = linalg.fill ins(%cst_0 : f32) outs(%26 : tensor<256xf32>) -> tensor<256xf32>
    %reduced = linalg.reduce ins(%25 : tensor<256x2048xf32>) outs(%27 : tensor<256xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %46 = arith.addf %in, %init : f32
        linalg.yield %46 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [256, 1] : tensor<256xf32> into tensor<256x1xf32>
    %28 = tensor.empty() : tensor<256x1xf32>
    %29 = linalg.fill ins(%cst_5 : f32) outs(%28 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %30 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded, %29 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%28 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %31 = tensor.empty() : tensor<256x1xf16>
    %32 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%30 : tensor<256x1xf32>) outs(%31 : tensor<256x1xf16>) -> tensor<256x1xf16>
    %collapsed_8 = tensor.collapse_shape %32 [[0, 1]] : tensor<256x1xf16> into tensor<256xf16>
    %broadcasted_9 = linalg.broadcast ins(%collapsed_8 : tensor<256xf16>) outs(%0 : tensor<256x2048xf16>) dimensions = [1] 
    %33 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_9, %4 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %34 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%24, %33 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %35 = arith.truncf %cst_2 : f64 to f32
    %36 = linalg.fill ins(%35 : f32) outs(%14 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %37 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%36 : tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %38 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%34, %37 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %39 = arith.truncf %cst_1 : f64 to f32
    %40 = linalg.fill ins(%39 : f32) outs(%14 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %41 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%40 : tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %42 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%24, %41 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %43 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%42, %4 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %44 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%38, %43 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %45 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%44 : tensor<256x2048xf16>) outs(%14 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    return %45 : tensor<256x2048xf32>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': False, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'cfx24j63t74q7xkzdz4jr3id56sfhn2alif4lk3gfvyjddxrrtgk', 'num_call_functions': 2, 'signature': {0: '*f16', 1: '*f16', 2: '*f16', 3: '*f16', 4: '*f32'}, 'ranks': [2, 2, 2, 2, 2], 'kernel_hash': 'f3bc08f17c26981f95364739f6159df67f5ee652886a9c49899923ff548f44ff'})
torch_npu.npu.set_device(0)


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    buf1 = empty_strided((256, 2048), (2048, 1), device='npu', dtype=torch.float32)
    stream0 = get_raw_stream(0)
    mlir_fused__to_copy_add_mean_mul_relu_sub_4.run(arg0_1, arg1_1, arg2_1, arg3_1, buf1, stream=stream0)
    del arg0_1
    del arg1_1
    del arg2_1
    del arg3_1
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((256, 2048), (2048, 1), device='npu:0', dtype=torch.float16)
    arg1_1 = rand_strided((256, 2048), (2048, 1), device='npu:0', dtype=torch.float16)
    arg2_1 = rand_strided((1, 2048), (2048, 1), device='npu:0', dtype=torch.float16)
    arg3_1 = rand_strided((256, 1), (1, 1), device='npu:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
