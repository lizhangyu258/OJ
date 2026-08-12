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


mlir_fused__to_copy_add_mul_relu_sigmoid_31 = async_compile.mlir_auto_fallback('mlir_fused__to_copy_add_mul_relu_sigmoid_31', '''
module {
  func.func @mlir_fused__to_copy_add_mul_relu_sigmoid_31(%arg0: tensor<?x?x?xf16>, %arg1: tensor<?x?x?xf16>, %arg2: tensor<1x1x?xf16>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: tensor<?x?x1xf16>) -> tensor<?x?x?xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %c1_i64 = arith.constant 1 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 2.500000e-01 : f32
    %cst_2 = arith.constant 1.250000e-01 : f32
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf16>
    %dim_3 = tensor.dim %arg0, %c1 : tensor<?x?x?xf16>
    %dim_4 = tensor.dim %arg0, %c2 : tensor<?x?x?xf16>
    %0 = tensor.empty(%dim, %dim_3, %dim_4) : tensor<?x?x?xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<?x?x?xf16>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dim_5 = tensor.dim %arg1, %c0 : tensor<?x?x?xf16>
    %dim_6 = tensor.dim %arg1, %c1 : tensor<?x?x?xf16>
    %dim_7 = tensor.dim %arg1, %c2 : tensor<?x?x?xf16>
    %2 = tensor.empty(%dim_5, %dim_6, %dim_7) : tensor<?x?x?xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<?x?x?xf16>) outs(%2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %4 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%3, %4 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %6 = tensor.empty(%dim, %dim_3, %dim_4) : tensor<?x?x?xi64>
    %7 = linalg.fill ins(%c1_i64 : i64) outs(%6 : tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
    %8 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%7 : tensor<?x?x?xi64>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%5, %8 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %9 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dim_8 = tensor.dim %arg2, %c2 : tensor<1x1x?xf16>
    %11 = tensor.empty(%dim_8) : tensor<1x1x?xf32>
    %12 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg2 : tensor<1x1x?xf16>) outs(%11 : tensor<1x1x?xf32>) -> tensor<1x1x?xf32>
    %13 = arith.index_cast %arg3 : i64 to index
    %14 = arith.index_cast %arg4 : i64 to index
    %15 = tensor.empty(%13, %14, %dim_8) : tensor<?x?x?xf32>
    %collapsed = tensor.collapse_shape %12 [[0, 1, 2]] : tensor<1x1x?xf32> into tensor<?xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<?xf32>) outs(%15 : tensor<?x?x?xf32>) dimensions = [0, 1] 
    %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %8 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%10, %16 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dim_9 = tensor.dim %arg6, %c0 : tensor<?x?x1xf16>
    %dim_10 = tensor.dim %arg6, %c1 : tensor<?x?x1xf16>
    %18 = tensor.empty(%dim_9, %dim_10) : tensor<?x?x1xf32>
    %19 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg6 : tensor<?x?x1xf16>) outs(%18 : tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %20 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%19 : tensor<?x?x1xf32>) outs(%18 : tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %21 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%20 : tensor<?x?x1xf32>) outs(%18 : tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%21, %cst : tensor<?x?x1xf32>, f32) outs(%18 : tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %22 : f32, tensor<?x?x1xf32>) outs(%18 : tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %24 = arith.index_cast %arg5 : i64 to index
    %25 = tensor.empty(%dim_9, %dim_10, %24) : tensor<?x?x?xf32>
    %collapsed_11 = tensor.collapse_shape %23 [[0], [1, 2]] : tensor<?x?x1xf32> into tensor<?x?xf32>
    %broadcasted_12 = linalg.broadcast ins(%collapsed_11 : tensor<?x?xf32>) outs(%25 : tensor<?x?x?xf32>) dimensions = [2] 
    %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%17, %broadcasted_12 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %27 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} ins(%26 : tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %28 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %29 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %28 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %30 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%29, %8 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %31 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%27, %30 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    return %31 : tensor<?x?x?xf32>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': True, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'cm6kfrnv5hgtrv7cpqfkb5vulv54ajgefczhp4g3s7ypswv4goi5', 'num_call_functions': 2, 'signature': {0: '*f16', 1: '*f16', 2: '*f16', 3: '*!torch.int', 4: '*!torch.int', 5: '*!torch.int', 6: '*f16', 7: '*f32'}, 'ranks': [3, 3, 3, 1, 1, 1, 3, 3], 'kernel_hash': '65fdd28441963d142d9738548b94dd650e449092b032b683ea4789640b3850a7'})
torch_npu.npu.set_device(0)


mlir_fused__to_copy_add_mean_mul_pow_relu_sigmoid_sub_32 = async_compile.mlir_auto_fallback('mlir_fused__to_copy_add_mean_mul_pow_relu_sigmoid_sub_32', '''
module {
  func.func @mlir_fused__to_copy_add_mean_mul_pow_relu_sigmoid_sub_32(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x1x?xf32>, %arg2: i64, %arg3: i64, %arg4: i64) -> tensor<?x?x?xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg1, %c0 : tensor<?x1x?xf32>
    %dim_0 = tensor.dim %arg1, %c2 : tensor<?x1x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x1x?xi64>
    %1 = linalg.fill ins(%arg2 : i64) outs(%0 : tensor<?x1x?xi64>) -> tensor<?x1x?xi64>
    %2 = tensor.empty(%dim, %dim_0) : tensor<?x1x?xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%1 : tensor<?x1x?xi64>) outs(%2 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg1, %3 : tensor<?x1x?xf32>, tensor<?x1x?xf32>) outs(%2 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %5 = arith.index_cast %arg2 : i64 to index
    %6 = tensor.empty(%dim, %5, %dim_0) : tensor<?x?x?xf32>
    %collapsed = tensor.collapse_shape %4 [[0, 1], [2]] : tensor<?x1x?xf32> into tensor<?x?xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<?x?xf32>) outs(%6 : tensor<?x?x?xf32>) dimensions = [1] 
    %dim_1 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
    %dim_2 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
    %dim_3 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
    %7 = tensor.empty(%dim_1, %dim_2, %dim_3) : tensor<?x?x?xi64>
    %8 = linalg.fill ins(%c1_i64 : i64) outs(%7 : tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
    %9 = tensor.empty(%dim_1, %dim_2, %dim_3) : tensor<?x?x?xf32>
    %10 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%8 : tensor<?x?x?xi64>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %10 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %11 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %13 = linalg.fill ins(%c2_i64 : i64) outs(%7 : tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
    %14 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%13 : tensor<?x?x?xi64>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %15 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%12, %14 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    return %15 : tensor<?x?x?xf32>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': True, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'ciq7sz2qs4kasrjrvq3zhlheu7p5kypwpz22lsg4mk7d5z3755js', 'num_call_functions': 2, 'signature': {0: '*f32', 1: '*f32', 2: '*!torch.int', 3: '*!torch.int', 4: '*!torch.int', 5: '*f32'}, 'ranks': [3, 3, 1, 1, 1, 3], 'kernel_hash': 'b26f1555d77dbcde3acfeb567026c0de9318ddcdbbc72a16f3f02fa7aaf47f1c'})


mlir_fused__to_copy_add_mean_mul_pow_relu_rsqrt_sigmoid_sub_33 = async_compile.mlir_auto_fallback('mlir_fused__to_copy_add_mean_mul_pow_relu_rsqrt_sigmoid_sub_33', '''
module {
  func.func @mlir_fused__to_copy_add_mean_mul_pow_relu_rsqrt_sigmoid_sub_33(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x1x?xf32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: tensor<?x1x?xf32>) -> tensor<?x?x?xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %c1_i64 = arith.constant 1 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 1.000000e-04 : f64
    %cst_0 = arith.constant 7.500000e-01 : f32
    %cst_1 = arith.constant 2.500000e-01 : f32
    %dim = tensor.dim %arg1, %c0 : tensor<?x1x?xf32>
    %dim_2 = tensor.dim %arg1, %c2 : tensor<?x1x?xf32>
    %0 = tensor.empty(%dim, %dim_2) : tensor<?x1x?xi64>
    %1 = linalg.fill ins(%arg2 : i64) outs(%0 : tensor<?x1x?xi64>) -> tensor<?x1x?xi64>
    %2 = tensor.empty(%dim, %dim_2) : tensor<?x1x?xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%1 : tensor<?x1x?xi64>) outs(%2 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg1, %3 : tensor<?x1x?xf32>, tensor<?x1x?xf32>) outs(%2 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %5 = arith.index_cast %arg2 : i64 to index
    %6 = tensor.empty(%dim, %5, %dim_2) : tensor<?x?x?xf32>
    %collapsed = tensor.collapse_shape %4 [[0, 1], [2]] : tensor<?x1x?xf32> into tensor<?x?xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<?x?xf32>) outs(%6 : tensor<?x?x?xf32>) dimensions = [1] 
    %dim_3 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
    %dim_4 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
    %dim_5 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
    %7 = tensor.empty(%dim_3, %dim_4, %dim_5) : tensor<?x?x?xi64>
    %8 = linalg.fill ins(%c1_i64 : i64) outs(%7 : tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
    %9 = tensor.empty(%dim_3, %dim_4, %dim_5) : tensor<?x?x?xf32>
    %10 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%8 : tensor<?x?x?xi64>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %10 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %11 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dim_6 = tensor.dim %arg5, %c0 : tensor<?x1x?xf32>
    %dim_7 = tensor.dim %arg5, %c2 : tensor<?x1x?xf32>
    %13 = tensor.empty(%dim_6, %dim_7) : tensor<?x1x?xi64>
    %14 = linalg.fill ins(%arg2 : i64) outs(%13 : tensor<?x1x?xi64>) -> tensor<?x1x?xi64>
    %15 = tensor.empty(%dim_6, %dim_7) : tensor<?x1x?xf32>
    %16 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%14 : tensor<?x1x?xi64>) outs(%15 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg5, %16 : tensor<?x1x?xf32>, tensor<?x1x?xf32>) outs(%15 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %18 = arith.truncf %cst : f64 to f32
    %19 = linalg.fill ins(%18 : f32) outs(%15 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %20 = linalg.fill ins(%c1_i64 : i64) outs(%13 : tensor<?x1x?xi64>) -> tensor<?x1x?xi64>
    %21 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%20 : tensor<?x1x?xi64>) outs(%15 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%19, %21 : tensor<?x1x?xf32>, tensor<?x1x?xf32>) outs(%15 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%17, %22 : tensor<?x1x?xf32>, tensor<?x1x?xf32>) outs(%15 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %24 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%23 : tensor<?x1x?xf32>) outs(%15 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %25 = tensor.empty(%dim_6, %5, %dim_7) : tensor<?x?x?xf32>
    %collapsed_8 = tensor.collapse_shape %24 [[0, 1], [2]] : tensor<?x1x?xf32> into tensor<?x?xf32>
    %broadcasted_9 = linalg.broadcast ins(%collapsed_8 : tensor<?x?xf32>) outs(%25 : tensor<?x?x?xf32>) dimensions = [1] 
    %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%12, %broadcasted_9 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %27 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%26, %27 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %29 = linalg.fill ins(%cst_1 : f32) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %30 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %29 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %31 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%30, %10 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %32 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%28, %31 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    return %32 : tensor<?x?x?xf32>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': True, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'c6tjinw4ubtjay652a3f764zbgfuf7z7wbayxdoehx72axpmtwqt', 'num_call_functions': 2, 'signature': {0: '*f32', 1: '*f32', 2: '*!torch.int', 3: '*!torch.int', 4: '*!torch.int', 5: '*f32', 6: '*f32'}, 'ranks': [3, 3, 1, 1, 1, 3, 3], 'kernel_hash': '7f3bc143b6e4fbc61f62661d143ee07e32a42eabd5267b81f022a54de1257712'})


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    buf0 = empty_strided((s0, s1, s2), (s1*s2, s2, 1), device='npu', dtype=torch.float32)
    stream0 = get_raw_stream(0)
    mlir_fused__to_copy_add_mul_relu_sigmoid_31.run(arg3_1, arg4_1, arg6_1, s0, s1, s2, arg5_1, buf0, stream=stream0)
    del arg3_1
    del arg4_1
    del arg5_1
    del arg6_1
    # Topologically Sorted Source Nodes: [x_fp32, residual_fp32, mul, add, bias_fp32, add_1, gate_fp32, sigmoid, gated, relu, mul_2, activated, seq_mean], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.sigmoid, aten.relu, aten.mean]
    buf1 = torch.ops.aten.sum.dim_IntList(buf0, [1], True, dtype=None)
    buf2 = buf1
    del buf1
    buf3 = empty_strided((s0, s1, s2), (s1*s2, s2, 1), device='npu', dtype=torch.float32)
    mlir_fused__to_copy_add_mean_mul_pow_relu_sigmoid_sub_32.run(buf0, buf2, s1, s0, s2, buf3, stream=stream0)
    # Topologically Sorted Source Nodes: [x_fp32, residual_fp32, mul, add, bias_fp32, add_1, gate_fp32, sigmoid, gated, relu, mul_2, activated, seq_mean, centered, square, mean_1], Original ATen: [aten._to_copy, aten.mul, aten.add, aten.sigmoid, aten.relu, aten.mean, aten.sub, aten.pow]
    buf4 = torch.ops.aten.sum.dim_IntList(buf3, [1], True, dtype=None)
    buf5 = buf4
    del buf4
    buf6 = buf3; del buf3  # reuse
    mlir_fused__to_copy_add_mean_mul_pow_relu_rsqrt_sigmoid_sub_33.run(buf0, buf2, s1, s0, s2, buf5, buf6, stream=stream0)
    return (buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 16
    arg1_1 = 128
    arg2_1 = 512
    arg3_1 = rand_strided((16, 128, 512), (65536, 512, 1), device='npu:0', dtype=torch.float16)
    arg4_1 = rand_strided((16, 128, 512), (65536, 512, 1), device='npu:0', dtype=torch.float16)
    arg5_1 = rand_strided((16, 128, 1), (128, 1, 1), device='npu:0', dtype=torch.float16)
    arg6_1 = rand_strided((1, 1, 512), (512, 512, 1), device='npu:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
