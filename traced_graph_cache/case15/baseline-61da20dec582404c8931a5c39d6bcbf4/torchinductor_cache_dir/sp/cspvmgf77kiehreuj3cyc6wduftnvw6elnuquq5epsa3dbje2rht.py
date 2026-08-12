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


mlir_fused_add_mean_mul_sub_sum_tanh_35 = async_compile.mlir_auto_fallback('mlir_fused_add_mean_mul_sub_sum_tanh_35', '''
module {
  func.func @mlir_fused_add_mean_mul_sub_sum_tanh_35(%arg0: tensor<32x128x256xf32>, %arg1: tensor<256xf32>) -> tensor<32x128x256xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 2.560000e+02 : f32
    %cst_2 = arith.constant 1.250000e-01 : f32
    %cst_3 = arith.constant 3.125000e-02 : f32
    %cst_4 = arith.constant 7.500000e-01 : f32
    %cst_5 = arith.constant 5.000000e-01 : f32
    %cst_6 = arith.constant 8.750000e-01 : f32
    %0 = tensor.empty() : tensor<32x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %reduced = linalg.reduce ins(%arg0 : tensor<32x128x256xf32>) outs(%1 : tensor<32x128xf32>) dimensions = [2] 
      (%in: f32, %init: f32) {
        %48 = arith.addf %in, %init : f32
        linalg.yield %48 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0], [1, 2]] output_shape [32, 128, 1] : tensor<32x128xf32> into tensor<32x128x1xf32>
    %2 = tensor.empty() : tensor<32x128x1xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<32x128x1xf32>) -> tensor<32x128x1xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded, %3 : tensor<32x128x1xf32>, tensor<32x128x1xf32>) outs(%2 : tensor<32x128x1xf32>) -> tensor<32x128x1xf32>
    %5 = tensor.empty() : tensor<32x128x256xf32>
    %collapsed = tensor.collapse_shape %4 [[0], [1, 2]] : tensor<32x128x1xf32> into tensor<32x128xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<32x128xf32>) outs(%5 : tensor<32x128x256xf32>) dimensions = [2] 
    %6 = tensor.empty() : tensor<32x128x256xi64>
    %7 = linalg.fill ins(%c1_i64 : i64) outs(%6 : tensor<32x128x256xi64>) -> tensor<32x128x256xi64>
    %8 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%7 : tensor<32x128x256xi64>) outs(%5 : tensor<32x128x256xf32>) -> tensor<32x128x256xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %8 : tensor<32x128x256xf32>, tensor<32x128x256xf32>) outs(%5 : tensor<32x128x256xf32>) -> tensor<32x128x256xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %9 : tensor<32x128x256xf32>, tensor<32x128x256xf32>) outs(%5 : tensor<32x128x256xf32>) -> tensor<32x128x256xf32>
    %reduced_7 = linalg.reduce ins(%10 : tensor<32x128x256xf32>) outs(%1 : tensor<32x128xf32>) dimensions = [2] 
      (%in: f32, %init: f32) {
        %48 = arith.addf %in, %init : f32
        linalg.yield %48 : f32
      }
    %expanded_8 = tensor.expand_shape %reduced_7 [[0], [1, 2]] output_shape [32, 128, 1] : tensor<32x128xf32> into tensor<32x128x1xf32>
    %11 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<32x128x1xf32>) -> tensor<32x128x1xf32>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded_8, %11 : tensor<32x128x1xf32>, tensor<32x128x1xf32>) outs(%2 : tensor<32x128x1xf32>) -> tensor<32x128x1xf32>
    %13 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tanh>} ins(%expanded_8 : tensor<32x128x1xf32>) outs(%2 : tensor<32x128x1xf32>) -> tensor<32x128x1xf32>
    %14 = linalg.fill ins(%cst_3 : f32) outs(%2 : tensor<32x128x1xf32>) -> tensor<32x128x1xf32>
    %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%13, %14 : tensor<32x128x1xf32>, tensor<32x128x1xf32>) outs(%2 : tensor<32x128x1xf32>) -> tensor<32x128x1xf32>
    %16 = tensor.empty() : tensor<32x128x1xi64>
    %17 = linalg.fill ins(%c1_i64 : i64) outs(%16 : tensor<32x128x1xi64>) -> tensor<32x128x1xi64>
    %18 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%17 : tensor<32x128x1xi64>) outs(%2 : tensor<32x128x1xf32>) -> tensor<32x128x1xf32>
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%15, %18 : tensor<32x128x1xf32>, tensor<32x128x1xf32>) outs(%2 : tensor<32x128x1xf32>) -> tensor<32x128x1xf32>
    %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%12, %19 : tensor<32x128x1xf32>, tensor<32x128x1xf32>) outs(%2 : tensor<32x128x1xf32>) -> tensor<32x128x1xf32>
    %collapsed_9 = tensor.collapse_shape %20 [[0], [1, 2]] : tensor<32x128x1xf32> into tensor<32x128xf32>
    %broadcasted_10 = linalg.broadcast ins(%collapsed_9 : tensor<32x128xf32>) outs(%5 : tensor<32x128x256xf32>) dimensions = [2] 
    %21 = tensor.empty() : tensor<256xf32>
    %22 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg1 : tensor<256xf32>) outs(%21 : tensor<256xf32>) -> tensor<256xf32>
    %23 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%22 : tensor<256xf32>) outs(%21 : tensor<256xf32>) -> tensor<256xf32>
    %24 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%23, %cst_0 : tensor<256xf32>, f32) outs(%21 : tensor<256xf32>) -> tensor<256xf32>
    %25 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst_0, %24 : f32, tensor<256xf32>) outs(%21 : tensor<256xf32>) -> tensor<256xf32>
    %expanded_11 = tensor.expand_shape %25 [[0, 1, 2]] output_shape [1, 1, 256] : tensor<256xf32> into tensor<1x1x256xf32>
    %26 = tensor.empty() : tensor<1x1x256xf32>
    %27 = linalg.fill ins(%cst_4 : f32) outs(%26 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
    %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded_11, %27 : tensor<1x1x256xf32>, tensor<1x1x256xf32>) outs(%26 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
    %29 = linalg.fill ins(%cst_5 : f32) outs(%26 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
    %30 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded_11, %29 : tensor<1x1x256xf32>, tensor<1x1x256xf32>) outs(%26 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
    %31 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tanh>} ins(%30 : tensor<1x1x256xf32>) outs(%26 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
    %32 = linalg.fill ins(%cst_2 : f32) outs(%26 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
    %33 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%31, %32 : tensor<1x1x256xf32>, tensor<1x1x256xf32>) outs(%26 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
    %34 = tensor.empty() : tensor<1x1x256xi64>
    %35 = linalg.fill ins(%c1_i64 : i64) outs(%34 : tensor<1x1x256xi64>) -> tensor<1x1x256xi64>
    %36 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%35 : tensor<1x1x256xi64>) outs(%26 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
    %37 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%33, %36 : tensor<1x1x256xf32>, tensor<1x1x256xf32>) outs(%26 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
    %38 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%28, %37 : tensor<1x1x256xf32>, tensor<1x1x256xf32>) outs(%26 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
    %collapsed_12 = tensor.collapse_shape %38 [[0, 1, 2]] : tensor<1x1x256xf32> into tensor<256xf32>
    %broadcasted_13 = linalg.broadcast ins(%collapsed_12 : tensor<256xf32>) outs(%5 : tensor<32x128x256xf32>) dimensions = [0, 1] 
    %39 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_13, %8 : tensor<32x128x256xf32>, tensor<32x128x256xf32>) outs(%5 : tensor<32x128x256xf32>) -> tensor<32x128x256xf32>
    %40 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%broadcasted_10, %39 : tensor<32x128x256xf32>, tensor<32x128x256xf32>) outs(%5 : tensor<32x128x256xf32>) -> tensor<32x128x256xf32>
    %41 = linalg.fill ins(%cst_6 : f32) outs(%5 : tensor<32x128x256xf32>) -> tensor<32x128x256xf32>
    %42 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%40, %41 : tensor<32x128x256xf32>, tensor<32x128x256xf32>) outs(%5 : tensor<32x128x256xf32>) -> tensor<32x128x256xf32>
    %43 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tanh>} ins(%40 : tensor<32x128x256xf32>) outs(%5 : tensor<32x128x256xf32>) -> tensor<32x128x256xf32>
    %44 = linalg.fill ins(%cst_2 : f32) outs(%5 : tensor<32x128x256xf32>) -> tensor<32x128x256xf32>
    %45 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%43, %44 : tensor<32x128x256xf32>, tensor<32x128x256xf32>) outs(%5 : tensor<32x128x256xf32>) -> tensor<32x128x256xf32>
    %46 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%45, %8 : tensor<32x128x256xf32>, tensor<32x128x256xf32>) outs(%5 : tensor<32x128x256xf32>) -> tensor<32x128x256xf32>
    %47 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%42, %46 : tensor<32x128x256xf32>, tensor<32x128x256xf32>) outs(%5 : tensor<32x128x256xf32>) -> tensor<32x128x256xf32>
    return %47 : tensor<32x128x256xf32>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': False, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'cvdits4mlc3jbvikhvtbgga6q5kfczwmy2gbhqhulyx7ypti4syf', 'num_call_functions': 2, 'signature': {0: '*f32', 1: '*f32', 2: '*f32'}, 'ranks': [3, 1, 3], 'kernel_hash': '2951f30c74dc7ef10b9c5f8ef62b005229eb72e902484d655c87408ab6fc9fb1'})
torch_npu.npu.set_device(0)


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    buf2 = empty_strided((32, 128, 256), (32768, 256, 1), device='npu', dtype=torch.float32)
    stream0 = get_raw_stream(0)
    mlir_fused_add_mean_mul_sub_sum_tanh_35.run(arg0_1, arg1_1, buf2, stream=stream0)
    del arg0_1
    del arg1_1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 128, 256), (32768, 256, 1), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((256, ), (1, ), device='npu:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
