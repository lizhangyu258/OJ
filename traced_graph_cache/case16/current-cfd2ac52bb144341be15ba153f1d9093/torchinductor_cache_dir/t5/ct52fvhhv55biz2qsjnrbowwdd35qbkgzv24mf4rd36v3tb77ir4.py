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


mlir_fused_add_exp_mul_permute_relu_sum_36 = async_compile.mlir_auto_fallback('mlir_fused_add_exp_mul_permute_relu_sum_36', '''
module {
  func.func @mlir_fused_add_exp_mul_permute_relu_sum_36(%arg0: tensor<32x128x64x16xf32>, %arg1: tensor<16xf32>, %arg2: tensor<32x128x64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<32x128x64xf32>, %arg5: tensor<128xf32>) -> tensor<32x128x64xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.250000e-01 : f32
    %cst_2 = arith.constant 1.562500e-02 : f32
    %0 = tensor.empty() : tensor<32x64x128x16xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<32x128x64x16xf32>) outs(%0 : tensor<32x64x128x16xf32>) permutation = [0, 2, 1, 3] 
    %1 = tensor.empty() : tensor<16xf32>
    %2 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg1 : tensor<16xf32>) outs(%1 : tensor<16xf32>) -> tensor<16xf32>
    %3 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%2 : tensor<16xf32>) outs(%1 : tensor<16xf32>) -> tensor<16xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %cst : tensor<16xf32>, f32) outs(%1 : tensor<16xf32>) -> tensor<16xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %4 : f32, tensor<16xf32>) outs(%1 : tensor<16xf32>) -> tensor<16xf32>
    %broadcasted = linalg.broadcast ins(%5 : tensor<16xf32>) outs(%0 : tensor<32x64x128x16xf32>) dimensions = [0, 1, 2] 
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%transposed, %broadcasted : tensor<32x64x128x16xf32>, tensor<32x64x128x16xf32>) outs(%0 : tensor<32x64x128x16xf32>) -> tensor<32x64x128x16xf32>
    %7 = tensor.empty() : tensor<32x64x128xf32>
    %8 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %reduced = linalg.reduce ins(%6 : tensor<32x64x128x16xf32>) outs(%8 : tensor<32x64x128xf32>) dimensions = [3] 
      (%in: f32, %init: f32) {
        %47 = arith.addf %in, %init : f32
        linalg.yield %47 : f32
      }
    %transposed_3 = linalg.transpose ins(%arg2 : tensor<32x128x64xf32>) outs(%7 : tensor<32x64x128xf32>) permutation = [0, 2, 1] 
    %9 = tensor.empty() : tensor<64xf32>
    %10 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tanh>} ins(%arg3 : tensor<64xf32>) outs(%9 : tensor<64xf32>) -> tensor<64xf32>
    %11 = linalg.fill ins(%cst_1 : f32) outs(%9 : tensor<64xf32>) -> tensor<64xf32>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%10, %11 : tensor<64xf32>, tensor<64xf32>) outs(%9 : tensor<64xf32>) -> tensor<64xf32>
    %13 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64xf32>) -> tensor<64xf32>
    %14 = tensor.empty() : tensor<64xi64>
    %15 = linalg.fill ins(%c1_i64 : i64) outs(%14 : tensor<64xi64>) -> tensor<64xi64>
    %16 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%15 : tensor<64xi64>) outs(%9 : tensor<64xf32>) -> tensor<64xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%13, %16 : tensor<64xf32>, tensor<64xf32>) outs(%9 : tensor<64xf32>) -> tensor<64xf32>
    %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%12, %17 : tensor<64xf32>, tensor<64xf32>) outs(%9 : tensor<64xf32>) -> tensor<64xf32>
    %broadcasted_4 = linalg.broadcast ins(%18 : tensor<64xf32>) outs(%7 : tensor<32x64x128xf32>) dimensions = [0, 2] 
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%transposed_3, %broadcasted_4 : tensor<32x64x128xf32>, tensor<32x64x128xf32>) outs(%7 : tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %transposed_5 = linalg.transpose ins(%arg4 : tensor<32x128x64xf32>) outs(%7 : tensor<32x64x128xf32>) permutation = [0, 2, 1] 
    %20 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%arg3 : tensor<64xf32>) outs(%9 : tensor<64xf32>) -> tensor<64xf32>
    %21 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%20 : tensor<64xf32>) outs(%9 : tensor<64xf32>) -> tensor<64xf32>
    %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%21, %cst : tensor<64xf32>, f32) outs(%9 : tensor<64xf32>) -> tensor<64xf32>
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %22 : f32, tensor<64xf32>) outs(%9 : tensor<64xf32>) -> tensor<64xf32>
    %expanded = tensor.expand_shape %23 [[0, 1, 2]] output_shape [1, 64, 1] : tensor<64xf32> into tensor<1x64x1xf32>
    %broadcasted_6 = linalg.broadcast ins(%23 : tensor<64xf32>) outs(%7 : tensor<32x64x128xf32>) dimensions = [0, 2] 
    %24 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%transposed_5, %broadcasted_6 : tensor<32x64x128xf32>, tensor<32x64x128xf32>) outs(%7 : tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %25 = tensor.empty() : tensor<32x64x128xi64>
    %26 = linalg.fill ins(%c1_i64 : i64) outs(%25 : tensor<32x64x128xi64>) -> tensor<32x64x128xi64>
    %27 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%26 : tensor<32x64x128xi64>) outs(%7 : tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%24, %27 : tensor<32x64x128xf32>, tensor<32x64x128xf32>) outs(%7 : tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %29 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%19, %28 : tensor<32x64x128xf32>, tensor<32x64x128xf32>) outs(%7 : tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %broadcasted_7 = linalg.broadcast ins(%arg5 : tensor<128xf32>) outs(%7 : tensor<32x64x128xf32>) dimensions = [0, 1] 
    %30 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_7, %27 : tensor<32x64x128xf32>, tensor<32x64x128xf32>) outs(%7 : tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %31 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%29, %30 : tensor<32x64x128xf32>, tensor<32x64x128xf32>) outs(%7 : tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %32 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%31 : tensor<32x64x128xf32>) outs(%7 : tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %33 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%reduced, %27 : tensor<32x64x128xf32>, tensor<32x64x128xf32>) outs(%7 : tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %34 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%32, %33 : tensor<32x64x128xf32>, tensor<32x64x128xf32>) outs(%7 : tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %35 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} ins(%34 : tensor<32x64x128xf32>) outs(%7 : tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %36 = tensor.empty() : tensor<1x64x1xf32>
    %37 = linalg.fill ins(%cst_2 : f32) outs(%36 : tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %38 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%expanded, %37 : tensor<1x64x1xf32>, tensor<1x64x1xf32>) outs(%36 : tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %39 = linalg.fill ins(%cst : f32) outs(%36 : tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %40 = tensor.empty() : tensor<1x64x1xi64>
    %41 = linalg.fill ins(%c1_i64 : i64) outs(%40 : tensor<1x64x1xi64>) -> tensor<1x64x1xi64>
    %42 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%41 : tensor<1x64x1xi64>) outs(%36 : tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %43 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%39, %42 : tensor<1x64x1xf32>, tensor<1x64x1xf32>) outs(%36 : tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %44 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%38, %43 : tensor<1x64x1xf32>, tensor<1x64x1xf32>) outs(%36 : tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
    %collapsed = tensor.collapse_shape %44 [[0, 1, 2]] : tensor<1x64x1xf32> into tensor<64xf32>
    %broadcasted_8 = linalg.broadcast ins(%collapsed : tensor<64xf32>) outs(%7 : tensor<32x64x128xf32>) dimensions = [0, 2] 
    %45 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%35, %broadcasted_8 : tensor<32x64x128xf32>, tensor<32x64x128xf32>) outs(%7 : tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %46 = tensor.empty() : tensor<32x128x64xf32>
    %transposed_9 = linalg.transpose ins(%45 : tensor<32x64x128xf32>) outs(%46 : tensor<32x128x64xf32>) permutation = [0, 2, 1] 
    return %transposed_9 : tensor<32x128x64xf32>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': False, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'c3ex66zro22754lthetx3cr62av2i2q3mhy55bhanlnjeznmmxpj', 'num_call_functions': 2, 'signature': {0: '*f32', 1: '*f32', 2: '*f32', 3: '*f32', 4: '*f32', 5: '*f32', 6: '*f32'}, 'ranks': [4, 1, 3, 1, 3, 1, 3], 'kernel_hash': '2b228613a4d7695b3800f84bea69c439320292a67d7b4532d8b40872b95e4b6a'})
torch_npu.npu.set_device(0)


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    buf2 = empty_strided((32, 128, 64), (8192, 64, 1), device='npu', dtype=torch.float32)
    stream0 = get_raw_stream(0)
    mlir_fused_add_exp_mul_permute_relu_sum_36.run(arg5_1, arg4_1, arg2_1, arg0_1, arg3_1, arg1_1, buf2, stream=stream0)
    del arg0_1
    del arg1_1
    del arg2_1
    del arg3_1
    del arg4_1
    del arg5_1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, ), (1, ), device='npu:0', dtype=torch.float32)
    arg1_1 = rand_strided((128, ), (1, ), device='npu:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, 128, 64), (8192, 64, 1), device='npu:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, 128, 64), (8192, 64, 1), device='npu:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='npu:0', dtype=torch.float32)
    arg5_1 = rand_strided((32, 128, 64, 16), (131072, 1024, 16, 1), device='npu:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
