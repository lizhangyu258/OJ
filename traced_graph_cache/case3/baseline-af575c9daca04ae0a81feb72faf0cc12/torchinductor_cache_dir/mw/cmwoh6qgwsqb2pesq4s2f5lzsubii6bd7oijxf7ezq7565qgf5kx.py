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
    def forward(self, arg0_1: "f16[1, 512]"):
        # No stacktrace found for following nodes
        _to_copy: "f32[1, 512]" = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float32);  arg0_1 = None
        return (_to_copy,)

"""
mlir_fused__to_copy_44 = async_compile.import_fx('mlir_fused__to_copy_44', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': False, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'cjcqg3xqil2v2ydd5tpmsz3p2qfr55ezuvcjpklvx6l7rwe2rgdd', 'num_call_functions': 1})
torch_npu.npu.set_device(0)


"""
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[128, 512]"):
        # No stacktrace found for following nodes
        _to_copy: "f32[128, 512]" = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float32);  arg0_1 = None
        return (_to_copy,)

"""
mlir_fused__to_copy_45 = async_compile.import_fx('mlir_fused__to_copy_45', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': False, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'c52rsgja4brm53fkf552guhu4r35hanfuqlik36ge7yl3wuangtw', 'num_call_functions': 1})


"""
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[512, 512]"):
        # No stacktrace found for following nodes
        _to_copy: "f32[512, 512]" = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float32);  arg0_1 = None
        return (_to_copy,)

"""
mlir_fused__to_copy_46 = async_compile.import_fx('mlir_fused__to_copy_46', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': False, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'cweyfcc7tert2rnymseplzrbhclpiq6nvtt56cyu7g7lvqt6sryg', 'num_call_functions': 1})


mlir_fused__to_copy_add_mean_mul_relu_47 = async_compile.mlir_auto_fallback('mlir_fused__to_copy_add_mean_mul_relu_47', '''
module {
  func.func @mlir_fused__to_copy_add_mean_mul_relu_47(%arg0: tensor<128x512xf32>, %arg1: tensor<128x1xf16>, %arg2: tensor<128x512xf16>) -> tensor<128x512xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e-02 : f64
    %cst_1 = arith.constant 5.000000e-02 : f64
    %cst_2 = arith.constant 5.120000e+02 : f32
    %0 = tensor.empty() : tensor<128x512xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} ins(%arg0 : tensor<128x512xf32>) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %2 = tensor.empty() : tensor<128x1xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<128x1xf16>) outs(%2 : tensor<128x1xf32>) -> tensor<128x1xf32>
    %collapsed = tensor.collapse_shape %3 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<128xf32>) outs(%0 : tensor<128x512xf32>) dimensions = [1] 
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %broadcasted : tensor<128x512xf32>, tensor<128x512xf32>) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg2 : tensor<128x512xf16>) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %6 = tensor.empty() : tensor<128x512xi64>
    %7 = linalg.fill ins(%c1_i64 : i64) outs(%6 : tensor<128x512xi64>) -> tensor<128x512xi64>
    %8 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%7 : tensor<128x512xi64>) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%5, %8 : tensor<128x512xf32>, tensor<128x512xf32>) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%4, %9 : tensor<128x512xf32>, tensor<128x512xf32>) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %11 = tensor.empty() : tensor<128xf32>
    %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<128xf32>) -> tensor<128xf32>
    %reduced = linalg.reduce ins(%10 : tensor<128x512xf32>) outs(%12 : tensor<128xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %25 = arith.addf %in, %init : f32
        linalg.yield %25 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
    %13 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<128x1xf32>) -> tensor<128x1xf32>
    %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded, %13 : tensor<128x1xf32>, tensor<128x1xf32>) outs(%2 : tensor<128x1xf32>) -> tensor<128x1xf32>
    %15 = arith.truncf %cst_1 : f64 to f32
    %16 = linalg.fill ins(%15 : f32) outs(%2 : tensor<128x1xf32>) -> tensor<128x1xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%14, %16 : tensor<128x1xf32>, tensor<128x1xf32>) outs(%2 : tensor<128x1xf32>) -> tensor<128x1xf32>
    %collapsed_3 = tensor.collapse_shape %17 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
    %broadcasted_4 = linalg.broadcast ins(%collapsed_3 : tensor<128xf32>) outs(%0 : tensor<128x512xf32>) dimensions = [1] 
    %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_4, %8 : tensor<128x512xf32>, tensor<128x512xf32>) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%10, %18 : tensor<128x512xf32>, tensor<128x512xf32>) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %20 = arith.truncf %cst_0 : f64 to f32
    %21 = linalg.fill ins(%20 : f32) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%5, %21 : tensor<128x512xf32>, tensor<128x512xf32>) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%22, %8 : tensor<128x512xf32>, tensor<128x512xf32>) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %24 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%19, %23 : tensor<128x512xf32>, tensor<128x512xf32>) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
    return %24 : tensor<128x512xf32>
  }
}
''', kernel_meta={'device_str': 'npu', 'device_index': 0, 'num_outputs': 1, 'non_contiguous_indices': {'inputs': [], 'outputs': []}, 'dynamic': False, 'mutated_indices': [], 'traced_graph_cache': 'traced_graph_cache', 'traced_graph_hash': 'cx263zj2cmb4ezmmplfnl2dvg67s6kvi42ufcuusj7i3qxoagmf7', 'num_call_functions': 2, 'signature': {0: '*f32', 1: '*f16', 2: '*f16', 3: '*f32'}, 'ranks': [2, 2, 2, 2], 'kernel_hash': 'a1e7b822b36375fb68f76fc966b1b237bd0298d62f546812aa6c824bc08bf69b'})


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    buf0 = empty_strided((1, 512), (512, 1), device='npu', dtype=torch.float32)
    stream0 = get_raw_stream(0)
    mlir_fused__to_copy_44.run(arg3_1, buf0, stream=stream0)
    del arg3_1
    buf1 = empty_strided((128, 512), (512, 1), device='npu', dtype=torch.float32)
    mlir_fused__to_copy_45.run(arg0_1, buf1, stream=stream0)
    del arg0_1
    buf2 = empty_strided((512, 512), (512, 1), device='npu', dtype=torch.float32)
    mlir_fused__to_copy_46.run(arg1_1, buf2, stream=stream0)
    del arg1_1
    # Topologically Sorted Source Nodes: [x_fp32, y_fp32, to_3], Original ATen: [aten._to_copy]
    buf3 = torch.ops.aten.addmm.default(buf0, buf1, buf2)
    del buf0
    del buf2
    buf4 = buf3
    del buf3
    buf6 = buf1; del buf1  # reuse
    mlir_fused__to_copy_add_mean_mul_relu_47.run(buf4, arg4_1, arg2_1, buf6, stream=stream0)
    del arg2_1
    del arg4_1
    return (buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((128, 512), (512, 1), device='npu:0', dtype=torch.float16)
    arg1_1 = rand_strided((512, 512), (512, 1), device='npu:0', dtype=torch.float16)
    arg2_1 = rand_strided((128, 512), (512, 1), device='npu:0', dtype=torch.float16)
    arg3_1 = rand_strided((1, 512), (512, 1), device='npu:0', dtype=torch.float16)
    arg4_1 = rand_strided((128, 1), (1, 1), device='npu:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
