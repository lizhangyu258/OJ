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


"""
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[1, s3]"):
        # No stacktrace found for following nodes
        _to_copy: "f32[1, s3]" = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float32);  arg0_1 = None
        return (_to_copy,)

"""
mlir_fused__to_copy_23 = async_compile.import_fx('mlir_fused__to_copy_23', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': True, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'cq6opihlzxz26pnvp3xj7he2vynxp537rkctl7wqrhhsvapyfmj5', 'num_call_functions': 1})
torch_npu.npu.set_device(0)


"""
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[s0, s1]"):
        # No stacktrace found for following nodes
        _to_copy: "f32[s0, s1]" = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float32);  arg0_1 = None
        return (_to_copy,)

"""
mlir_fused__to_copy_24 = async_compile.import_fx('mlir_fused__to_copy_24', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': True, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'c7qvusiq3t4ciupv3qj6izdpwdyrqfuyoadg4yjmetxapc4rmzd2', 'num_call_functions': 1})


mlir_fused__to_copy_add_mean_mul_relu_25 = async_compile.mlir_auto_fallback('mlir_fused__to_copy_add_mean_mul_relu_25', '''
module {
  func.func @mlir_fused__to_copy_add_mean_mul_relu_25(%arg0: tensor<?x?xf32>, %arg1: tensor<?x1xf16>, %arg2: i64, %arg3: i64) -> tensor<?x?xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %c1_i64 = arith.constant 1 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-02 : f64
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim, %dim_1) : tensor<?x?xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} ins(%arg0 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %dim_2 = tensor.dim %arg1, %c0 : tensor<?x1xf16>
    %2 = tensor.empty(%dim_2) : tensor<?x1xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<?x1xf16>) outs(%2 : tensor<?x1xf32>) -> tensor<?x1xf32>
    %4 = arith.index_cast %arg3 : i64 to index
    %5 = tensor.empty(%dim_2, %4) : tensor<?x?xf32>
    %collapsed = tensor.collapse_shape %3 [[0, 1]] : tensor<?x1xf32> into tensor<?xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<?xf32>) outs(%5 : tensor<?x?xf32>) dimensions = [1] 
    %6 = tensor.empty(%dim, %dim_1) : tensor<?x?xi64>
    %7 = linalg.fill ins(%c1_i64 : i64) outs(%6 : tensor<?x?xi64>) -> tensor<?x?xi64>
    %8 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%7 : tensor<?x?xi64>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %9 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %11 = tensor.empty(%dim) : tensor<?xf32>
    %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<?xf32>) -> tensor<?xf32>
    %reduced = linalg.reduce ins(%10 : tensor<?x?xf32>) outs(%12 : tensor<?xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %24 = arith.addf %in, %init : f32
        linalg.yield %24 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [%dim, 1] : tensor<?xf32> into tensor<?x1xf32>
    %13 = tensor.empty(%dim) : tensor<?x1xi64>
    %14 = linalg.fill ins(%arg3 : i64) outs(%13 : tensor<?x1xi64>) -> tensor<?x1xi64>
    %15 = tensor.empty(%dim) : tensor<?x1xf32>
    %16 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%14 : tensor<?x1xi64>) outs(%15 : tensor<?x1xf32>) -> tensor<?x1xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded, %16 : tensor<?x1xf32>, tensor<?x1xf32>) outs(%15 : tensor<?x1xf32>) -> tensor<?x1xf32>
    %18 = arith.truncf %cst_0 : f64 to f32
    %19 = linalg.fill ins(%18 : f32) outs(%15 : tensor<?x1xf32>) -> tensor<?x1xf32>
    %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%17, %19 : tensor<?x1xf32>, tensor<?x1xf32>) outs(%15 : tensor<?x1xf32>) -> tensor<?x1xf32>
    %21 = tensor.empty(%dim, %4) : tensor<?x?xf32>
    %collapsed_3 = tensor.collapse_shape %20 [[0, 1]] : tensor<?x1xf32> into tensor<?xf32>
    %broadcasted_4 = linalg.broadcast ins(%collapsed_3 : tensor<?xf32>) outs(%21 : tensor<?x?xf32>) dimensions = [1] 
    %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_4, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%10, %22 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %23 : tensor<?x?xf32>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': True, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'cvttrtwpecxu3nluhot4zns3ffojm6h4h342mgki55bhypoktfcz', 'num_call_functions': 2, 'signature': {0: '*f32', 1: '*f16', 2: '*!torch.int', 3: '*!torch.int', 4: '*f32'}, 'ranks': [2, 2, 1, 1, 2], 'kernel_hash': '6724227c6026885df21325ae96ceff2c02f64c051b4d974321d55c8a7d355b26'})


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s3 = arg3_1
    buf0 = empty_strided((1, s3), (s3, 1), device='npu', dtype=torch.float32)
    stream0 = get_raw_stream(0)
    mlir_fused__to_copy_23.run(arg5_1, buf0, stream=stream0)
    del arg5_1
    buf1 = empty_strided((s0, s1), (s1, 1), device='npu', dtype=torch.float32)
    mlir_fused__to_copy_24.run(arg2_1, buf1, stream=stream0)
    del arg2_1
    buf2 = empty_strided((s1, s3), (s3, 1), device='npu', dtype=torch.float32)
    mlir_fused__to_copy_24.run(arg4_1, buf2, stream=stream0)
    del arg4_1
    # Topologically Sorted Source Nodes: [x_fp32, weight_fp32, to_2], Original ATen: [aten._to_copy]
    buf3 = torch.ops.aten.addmm.default(buf0, buf1, buf2)
    del buf0
    del buf1
    del buf2
    buf4 = buf3
    del buf3
    buf6 = empty_strided((s0, s3), (s3, 1), device='npu', dtype=torch.float32)
    mlir_fused__to_copy_add_mean_mul_relu_25.run(buf4, arg6_1, s0, s3, buf6, stream=stream0)
    del arg6_1
    return (buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 96
    arg1_1 = 384
    arg2_1 = rand_strided((96, 384), (384, 1), device='npu:0', dtype=torch.float16)
    arg3_1 = 256
    arg4_1 = rand_strided((384, 256), (256, 1), device='npu:0', dtype=torch.float16)
    arg5_1 = rand_strided((1, 256), (256, 1), device='npu:0', dtype=torch.float16)
    arg6_1 = rand_strided((96, 1), (1, 1), device='npu:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
