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


mlir_fused__to_copy_mul_12 = async_compile.mlir_auto_fallback('mlir_fused__to_copy_mul_12', '''
module {
  func.func @mlir_fused__to_copy_mul_12(%arg0: tensor<4x8x128x64xf16>) -> tensor<4x8x128x64xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.35355339059327379 : f64
    %0 = tensor.empty() : tensor<4x8x128x64xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x8x128x64xf16>) outs(%0 : tensor<4x8x128x64xf32>) -> tensor<4x8x128x64xf32>
    %2 = arith.truncf %cst : f64 to f32
    %3 = linalg.fill ins(%2 : f32) outs(%0 : tensor<4x8x128x64xf32>) -> tensor<4x8x128x64xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %3 : tensor<4x8x128x64xf32>, tensor<4x8x128x64xf32>) outs(%0 : tensor<4x8x128x64xf32>) -> tensor<4x8x128x64xf32>
    return %4 : tensor<4x8x128x64xf32>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': False, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'cfc2ru6m3tfut6kvwe7jwc6twb7pop5hg6nhdkxxwos7y5boi3a5', 'num_call_functions': 2, 'signature': {0: '*f16', 1: '*f32'}, 'ranks': [4, 4], 'kernel_hash': '8a5a1ec0a139f115760ee24173e51e7cfb7e80e4730dc86b8084affed4acc2f5'})
torch_npu.npu.set_device(0)


mlir_fused_mul_13 = async_compile.mlir_auto_fallback('mlir_fused_mul_13', '''
module {
  func.func @mlir_fused_mul_13(%arg0: tensor<4x8x128x64xf16>) -> tensor<4x8x64x128xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.35355339059327379 : f64
    %0 = tensor.empty() : tensor<4x8x128x64xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x8x128x64xf16>) outs(%0 : tensor<4x8x128x64xf32>) -> tensor<4x8x128x64xf32>
    %2 = tensor.empty() : tensor<4x8x64x128xf32>
    %transposed = linalg.transpose ins(%1 : tensor<4x8x128x64xf32>) outs(%2 : tensor<4x8x64x128xf32>) permutation = [0, 1, 3, 2] 
    %3 = arith.truncf %cst : f64 to f32
    %4 = linalg.fill ins(%3 : f32) outs(%2 : tensor<4x8x64x128xf32>) -> tensor<4x8x64x128xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%transposed, %4 : tensor<4x8x64x128xf32>, tensor<4x8x64x128xf32>) outs(%2 : tensor<4x8x64x128xf32>) -> tensor<4x8x64x128xf32>
    return %5 : tensor<4x8x64x128xf32>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': False, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'ctgqtwzn773hsmgjy3sjlgi5uzjuegxuxkrqhfnh3soizrqjm6ea', 'num_call_functions': 2, 'signature': {0: '*f16', 1: '*f32'}, 'ranks': [4, 4], 'kernel_hash': '544dfd059dd0d48f388391c4f75add63ff97f315b1fc1685921c017b6247eb52'})


mlir_fused__safe_softmax_add_14 = async_compile.mlir_auto_fallback('mlir_fused__safe_softmax_add_14', '''
module {
  func.func @mlir_fused__safe_softmax_add_14(%arg0: tensor<32x128x128xf32>, %arg1: tensor<1x1x128x128xf16>) -> tensor<4x8x128x128xi1> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0xFF800000 : f32
    %expanded = tensor.expand_shape %arg0 [[0, 1], [2], [3]] output_shape [4, 8, 128, 128] : tensor<32x128x128xf32> into tensor<4x8x128x128xf32>
    %0 = tensor.empty() : tensor<1x1x128x128xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<1x1x128x128xf16>) outs(%0 : tensor<1x1x128x128xf32>) -> tensor<1x1x128x128xf32>
    %2 = tensor.empty() : tensor<4x8x128x128xf32>
    %collapsed = tensor.collapse_shape %1 [[0, 1, 2], [3]] : tensor<1x1x128x128xf32> into tensor<128x128xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<128x128xf32>) outs(%2 : tensor<4x8x128x128xf32>) dimensions = [0, 1] 
    %3 = tensor.empty() : tensor<4x8x128x128xi64>
    %4 = linalg.fill ins(%c1_i64 : i64) outs(%3 : tensor<4x8x128x128xi64>) -> tensor<4x8x128x128xi64>
    %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%4 : tensor<4x8x128x128xi64>) outs(%2 : tensor<4x8x128x128xf32>) -> tensor<4x8x128x128xf32>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %5 : tensor<4x8x128x128xf32>, tensor<4x8x128x128xf32>) outs(%2 : tensor<4x8x128x128xf32>) -> tensor<4x8x128x128xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%expanded, %6 : tensor<4x8x128x128xf32>, tensor<4x8x128x128xf32>) outs(%2 : tensor<4x8x128x128xf32>) -> tensor<4x8x128x128xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x8x128x128xf32>) -> tensor<4x8x128x128xf32>
    %9 = tensor.empty() : tensor<4x8x128x128xi1>
    %10 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%7, %8 : tensor<4x8x128x128xf32>, tensor<4x8x128x128xf32>) outs(%9 : tensor<4x8x128x128xi1>) -> tensor<4x8x128x128xi1>
    return %10 : tensor<4x8x128x128xi1>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': False, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'cu237abqnsy2pal3hmjkjldh7l6mprqyiiaxgddlj7uaf3qmaj7x', 'num_call_functions': 2, 'signature': {0: '*f32', 1: '*f16', 2: '*i1'}, 'ranks': [3, 4, 4], 'kernel_hash': 'b7496c0bb59d2b488c961804c03aa07ebaa2109d109a0317dc0f887e7a97a512'})


mlir_fused__safe_softmax_add_15 = async_compile.mlir_auto_fallback('mlir_fused__safe_softmax_add_15', '''
module {
  func.func @mlir_fused__safe_softmax_add_15(%arg0: tensor<32x128x128xf32>, %arg1: tensor<1x1x128x128xf16>, %arg2: tensor<4x8x128x1xi1>) -> tensor<4x8x128x128xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant dense<0> : tensor<i64>
    %c1_i64 = arith.constant 1 : i64
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %expanded = tensor.expand_shape %arg0 [[0, 1], [2], [3]] output_shape [4, 8, 128, 128] : tensor<32x128x128xf32> into tensor<4x8x128x128xf32>
    %0 = tensor.empty() : tensor<1x1x128x128xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<1x1x128x128xf16>) outs(%0 : tensor<1x1x128x128xf32>) -> tensor<1x1x128x128xf32>
    %2 = tensor.empty() : tensor<4x8x128x128xf32>
    %collapsed = tensor.collapse_shape %1 [[0, 1, 2], [3]] : tensor<1x1x128x128xf32> into tensor<128x128xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<128x128xf32>) outs(%2 : tensor<4x8x128x128xf32>) dimensions = [0, 1] 
    %3 = tensor.empty() : tensor<4x8x128x128xi64>
    %4 = linalg.fill ins(%c1_i64 : i64) outs(%3 : tensor<4x8x128x128xi64>) -> tensor<4x8x128x128xi64>
    %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%4 : tensor<4x8x128x128xi64>) outs(%2 : tensor<4x8x128x128xf32>) -> tensor<4x8x128x128xf32>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %5 : tensor<4x8x128x128xf32>, tensor<4x8x128x128xf32>) outs(%2 : tensor<4x8x128x128xf32>) -> tensor<4x8x128x128xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%expanded, %6 : tensor<4x8x128x128xf32>, tensor<4x8x128x128xf32>) outs(%2 : tensor<4x8x128x128xf32>) -> tensor<4x8x128x128xf32>
    %8 = tensor.empty() : tensor<4x8x128xf32>
    %9 = linalg.fill ins(%cst_0 : f32) outs(%8 : tensor<4x8x128xf32>) -> tensor<4x8x128xf32>
    %reduced = linalg.reduce ins(%7 : tensor<4x8x128x128xf32>) outs(%9 : tensor<4x8x128xf32>) dimensions = [3] 
      (%in: f32, %init: f32) {
        %19 = arith.maximumf %in, %init : f32
        linalg.yield %19 : f32
      }
    %broadcasted_2 = linalg.broadcast ins(%reduced : tensor<4x8x128xf32>) outs(%2 : tensor<4x8x128x128xf32>) dimensions = [3] 
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_2, %5 : tensor<4x8x128x128xf32>, tensor<4x8x128x128xf32>) outs(%2 : tensor<4x8x128x128xf32>) -> tensor<4x8x128x128xf32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%7, %10 : tensor<4x8x128x128xf32>, tensor<4x8x128x128xf32>) outs(%2 : tensor<4x8x128x128xf32>) -> tensor<4x8x128x128xf32>
    %12 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%11 : tensor<4x8x128x128xf32>) outs(%2 : tensor<4x8x128x128xf32>) -> tensor<4x8x128x128xf32>
    %13 = linalg.fill ins(%cst_1 : f32) outs(%8 : tensor<4x8x128xf32>) -> tensor<4x8x128xf32>
    %reduced_3 = linalg.reduce ins(%12 : tensor<4x8x128x128xf32>) outs(%13 : tensor<4x8x128xf32>) dimensions = [3] 
      (%in: f32, %init: f32) {
        %19 = arith.addf %in, %init : f32
        linalg.yield %19 : f32
      }
    %14 = tensor.empty() : tensor<4x8x128x128xi1>
    %collapsed_4 = tensor.collapse_shape %arg2 [[0], [1], [2, 3]] : tensor<4x8x128x1xi1> into tensor<4x8x128xi1>
    %broadcasted_5 = linalg.broadcast ins(%collapsed_4 : tensor<4x8x128xi1>) outs(%14 : tensor<4x8x128x128xi1>) dimensions = [3] 
    %15 = tensor.empty() : tensor<f32>
    %16 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%cst : tensor<i64>) outs(%15 : tensor<f32>) -> tensor<f32>
    %broadcasted_6 = linalg.broadcast ins(%16 : tensor<f32>) outs(%2 : tensor<4x8x128x128xf32>) dimensions = [0, 1, 2, 3] 
    %broadcasted_7 = linalg.broadcast ins(%reduced_3 : tensor<4x8x128xf32>) outs(%2 : tensor<4x8x128x128xf32>) dimensions = [3] 
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%12, %broadcasted_7 : tensor<4x8x128x128xf32>, tensor<4x8x128x128xf32>) outs(%2 : tensor<4x8x128x128xf32>) -> tensor<4x8x128x128xf32>
    %18 = hfusion.select ins(%broadcasted_5, %broadcasted_6, %17 : tensor<4x8x128x128xi1>, tensor<4x8x128x128xf32>, tensor<4x8x128x128xf32>) outs(%2 : tensor<4x8x128x128xf32>) -> tensor<4x8x128x128xf32>
    return %18 : tensor<4x8x128x128xf32>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': False, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'cmowtc5hylfbjx4wh5yhbz6lthcijwiuxzz4h55tvq2qmzl7powe', 'num_call_functions': 2, 'signature': {0: '*f32', 1: '*f16', 2: '*i1', 3: '*f32'}, 'ranks': [3, 4, 4, 4], 'kernel_hash': '385f2ba5e19df7cda28b42e666008a0d4423ddc225b03eeba4763f310fe36409'})


"""
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[4, 8, 128, 64]"):
        # No stacktrace found for following nodes
        _to_copy: "f32[4, 8, 128, 64]" = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float32);  arg0_1 = None
        return (_to_copy,)

"""
mlir_fused__to_copy_16 = async_compile.import_fx('mlir_fused__to_copy_16', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': False, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'cb4urixj5av7jjgp7sdj3tyuee57q42j4oz46op4u2qbcmfpe4so', 'num_call_functions': 1})


mlir_fused__to_copy_add_mean_mul_relu_17 = async_compile.mlir_auto_fallback('mlir_fused__to_copy_add_mean_mul_relu_17', '''
module {
  func.func @mlir_fused__to_copy_add_mean_mul_relu_17(%arg0: tensor<32x128x64xf32>, %arg1: tensor<4x8x128x64xf16>) -> tensor<4x8x128x64xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 5.000000e-02 : f64
    %cst_2 = arith.constant 1.000000e-01 : f64
    %cst_3 = arith.constant 7.500000e-01 : f32
    %cst_4 = arith.constant 2.500000e-01 : f32
    %cst_5 = arith.constant 6.400000e+01 : f32
    %expanded = tensor.expand_shape %arg0 [[0, 1], [2], [3]] output_shape [4, 8, 128, 64] : tensor<32x128x64xf32> into tensor<4x8x128x64xf32>
    %0 = tensor.empty() : tensor<4x8x128x64xf16>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%expanded : tensor<4x8x128x64xf32>) outs(%0 : tensor<4x8x128x64xf16>) -> tensor<4x8x128x64xf16>
    %2 = arith.truncf %cst_2 : f64 to f32
    %3 = tensor.empty() : tensor<4x8x128x64xf32>
    %4 = linalg.fill ins(%2 : f32) outs(%3 : tensor<4x8x128x64xf32>) -> tensor<4x8x128x64xf32>
    %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%4 : tensor<4x8x128x64xf32>) outs(%0 : tensor<4x8x128x64xf16>) -> tensor<4x8x128x64xf16>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %5 : tensor<4x8x128x64xf16>, tensor<4x8x128x64xf16>) outs(%0 : tensor<4x8x128x64xf16>) -> tensor<4x8x128x64xf16>
    %7 = tensor.empty() : tensor<4x8x128x64xi64>
    %8 = linalg.fill ins(%c1_i64 : i64) outs(%7 : tensor<4x8x128x64xi64>) -> tensor<4x8x128x64xi64>
    %9 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%8 : tensor<4x8x128x64xi64>) outs(%0 : tensor<4x8x128x64xf16>) -> tensor<4x8x128x64xf16>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %9 : tensor<4x8x128x64xf16>, tensor<4x8x128x64xf16>) outs(%0 : tensor<4x8x128x64xf16>) -> tensor<4x8x128x64xf16>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %10 : tensor<4x8x128x64xf16>, tensor<4x8x128x64xf16>) outs(%0 : tensor<4x8x128x64xf16>) -> tensor<4x8x128x64xf16>
    %12 = linalg.fill ins(%cst_3 : f32) outs(%3 : tensor<4x8x128x64xf32>) -> tensor<4x8x128x64xf32>
    %13 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%12 : tensor<4x8x128x64xf32>) outs(%0 : tensor<4x8x128x64xf16>) -> tensor<4x8x128x64xf16>
    %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%11, %13 : tensor<4x8x128x64xf16>, tensor<4x8x128x64xf16>) outs(%0 : tensor<4x8x128x64xf16>) -> tensor<4x8x128x64xf16>
    %15 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} ins(%1 : tensor<4x8x128x64xf16>) outs(%0 : tensor<4x8x128x64xf16>) -> tensor<4x8x128x64xf16>
    %16 = linalg.fill ins(%cst_4 : f32) outs(%3 : tensor<4x8x128x64xf32>) -> tensor<4x8x128x64xf32>
    %17 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%16 : tensor<4x8x128x64xf32>) outs(%0 : tensor<4x8x128x64xf16>) -> tensor<4x8x128x64xf16>
    %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%15, %17 : tensor<4x8x128x64xf16>, tensor<4x8x128x64xf16>) outs(%0 : tensor<4x8x128x64xf16>) -> tensor<4x8x128x64xf16>
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%18, %9 : tensor<4x8x128x64xf16>, tensor<4x8x128x64xf16>) outs(%0 : tensor<4x8x128x64xf16>) -> tensor<4x8x128x64xf16>
    %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%14, %19 : tensor<4x8x128x64xf16>, tensor<4x8x128x64xf16>) outs(%0 : tensor<4x8x128x64xf16>) -> tensor<4x8x128x64xf16>
    %21 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%20 : tensor<4x8x128x64xf16>) outs(%3 : tensor<4x8x128x64xf32>) -> tensor<4x8x128x64xf32>
    %22 = tensor.empty() : tensor<4x8x128xf32>
    %23 = linalg.fill ins(%cst_0 : f32) outs(%22 : tensor<4x8x128xf32>) -> tensor<4x8x128xf32>
    %reduced = linalg.reduce ins(%21 : tensor<4x8x128x64xf32>) outs(%23 : tensor<4x8x128xf32>) dimensions = [3] 
      (%in: f32, %init: f32) {
        %36 = arith.addf %in, %init : f32
        linalg.yield %36 : f32
      }
    %expanded_6 = tensor.expand_shape %reduced [[0], [1], [2, 3]] output_shape [4, 8, 128, 1] : tensor<4x8x128xf32> into tensor<4x8x128x1xf32>
    %24 = tensor.empty() : tensor<4x8x128x1xf32>
    %25 = linalg.fill ins(%cst_5 : f32) outs(%24 : tensor<4x8x128x1xf32>) -> tensor<4x8x128x1xf32>
    %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded_6, %25 : tensor<4x8x128x1xf32>, tensor<4x8x128x1xf32>) outs(%24 : tensor<4x8x128x1xf32>) -> tensor<4x8x128x1xf32>
    %27 = tensor.empty() : tensor<4x8x128x1xf16>
    %28 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%26 : tensor<4x8x128x1xf32>) outs(%27 : tensor<4x8x128x1xf16>) -> tensor<4x8x128x1xf16>
    %29 = arith.truncf %cst_1 : f64 to f32
    %30 = linalg.fill ins(%29 : f32) outs(%24 : tensor<4x8x128x1xf32>) -> tensor<4x8x128x1xf32>
    %31 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%30 : tensor<4x8x128x1xf32>) outs(%27 : tensor<4x8x128x1xf16>) -> tensor<4x8x128x1xf16>
    %32 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%28, %31 : tensor<4x8x128x1xf16>, tensor<4x8x128x1xf16>) outs(%27 : tensor<4x8x128x1xf16>) -> tensor<4x8x128x1xf16>
    %collapsed = tensor.collapse_shape %32 [[0], [1], [2, 3]] : tensor<4x8x128x1xf16> into tensor<4x8x128xf16>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<4x8x128xf16>) outs(%0 : tensor<4x8x128x64xf16>) dimensions = [3] 
    %33 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %9 : tensor<4x8x128x64xf16>, tensor<4x8x128x64xf16>) outs(%0 : tensor<4x8x128x64xf16>) -> tensor<4x8x128x64xf16>
    %34 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%20, %33 : tensor<4x8x128x64xf16>, tensor<4x8x128x64xf16>) outs(%0 : tensor<4x8x128x64xf16>) -> tensor<4x8x128x64xf16>
    %35 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%34 : tensor<4x8x128x64xf16>) outs(%3 : tensor<4x8x128x64xf32>) -> tensor<4x8x128x64xf32>
    return %35 : tensor<4x8x128x64xf32>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': False, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'c3r46x2ezk74cwnk6ztcfhe3tfvtszhhmgegpzunwisopjekovbr', 'num_call_functions': 2, 'signature': {0: '*f32', 1: '*f16', 2: '*f32'}, 'ranks': [3, 4, 4], 'kernel_hash': 'b10415f78d0c9cb9f8c099dce7753891c5e972acd36e52726bdb373bf6768d89'})


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    buf0 = empty_strided((4, 8, 128, 64), (65536, 8192, 64, 1), device='npu', dtype=torch.float32)
    stream0 = get_raw_stream(0)
    mlir_fused__to_copy_mul_12.run(arg3_1, buf0, stream=stream0)
    buf1 = empty_strided((4, 8, 64, 128), (65536, 8192, 128, 1), device='npu', dtype=torch.float32)
    mlir_fused_mul_13.run(arg2_1, buf1, stream=stream0)
    del arg2_1
    # Topologically Sorted Source Nodes: [context], Original ATen: [aten.bmm]
    buf2 = torch.ops.aten.bmm.default(reinterpret_tensor(buf0, (32, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf1, (32, 64, 128), (8192, 128, 1), 0))
    del buf0
    buf3 = buf2
    del buf2
    buf4 = empty_strided((4, 8, 128, 128), (131072, 16384, 128, 1), device='npu', dtype=torch.bool)
    mlir_fused__safe_softmax_add_14.run(buf3, arg0_1, buf4, stream=stream0)
    # Topologically Sorted Source Nodes: [context], Original ATen: [aten.add, aten._safe_softmax]
    buf5 = torch.ops.aten.logical_not.default(buf4)
    del buf4
    buf6 = buf5
    del buf5
    # Topologically Sorted Source Nodes: [context], Original ATen: [aten._safe_softmax]
    buf7 = torch.ops.aten.any.dim(buf6, -1, True)
    del buf6
    buf8 = buf7
    del buf7
    # Topologically Sorted Source Nodes: [context], Original ATen: [aten._safe_softmax]
    buf9 = torch.ops.aten.logical_not.default(buf8)
    del buf8
    buf10 = buf9
    del buf9
    buf13 = empty_strided((4, 8, 128, 128), (131072, 16384, 128, 1), device='npu', dtype=torch.float32)
    mlir_fused__safe_softmax_add_15.run(buf3, arg0_1, buf10, buf13, stream=stream0)
    del arg0_1
    del buf10
    del buf3
    buf14 = reinterpret_tensor(buf1, (4, 8, 128, 64), (65536, 8192, 64, 1), 0); del buf1  # reuse
    mlir_fused__to_copy_16.run(arg1_1, buf14, stream=stream0)
    del arg1_1
    # Topologically Sorted Source Nodes: [context], Original ATen: [aten.bmm]
    buf15 = torch.ops.aten.bmm.default(reinterpret_tensor(buf13, (32, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf14, (32, 128, 64), (8192, 64, 1), 0))
    del buf13
    buf16 = buf15
    del buf15
    buf18 = buf14; del buf14  # reuse
    mlir_fused__to_copy_add_mean_mul_relu_17.run(buf16, arg3_1, buf18, stream=stream0)
    del arg3_1
    return (buf18, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1, 128, 128), (16384, 16384, 128, 1), device='npu:0', dtype=torch.float16)
    arg1_1 = rand_strided((4, 8, 128, 64), (65536, 8192, 64, 1), device='npu:0', dtype=torch.float16)
    arg2_1 = rand_strided((4, 8, 128, 64), (65536, 8192, 64, 1), device='npu:0', dtype=torch.float16)
    arg3_1 = rand_strided((4, 8, 128, 64), (65536, 8192, 64, 1), device='npu:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
