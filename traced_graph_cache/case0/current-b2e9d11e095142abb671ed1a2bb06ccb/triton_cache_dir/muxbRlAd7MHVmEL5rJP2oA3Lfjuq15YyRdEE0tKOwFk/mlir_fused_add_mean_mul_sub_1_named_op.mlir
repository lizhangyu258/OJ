
module {
  func.func @mlir_fused_add_mean_mul_sub_1(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<f32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+03 : f32
    %cst_1 = arith.constant 1.600000e+01 : f32
    %0 = tensor.empty() : tensor<16xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg1, %1 : tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %3 = tensor.empty() : tensor<16xi64>
    %4 = linalg.fill ins(%c1_i64 : i64) outs(%3 : tensor<16xi64>) -> tensor<16xi64>
    %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%4 : tensor<16xi64>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %5 : tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %6 : tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    %8 = tensor.empty() : tensor<f32>
    %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<f32>) -> tensor<f32>
    %reduced = linalg.reduce ins(%7 : tensor<16xf32>) outs(%9 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %12 = arith.addf %in, %init : f32
        linalg.yield %12 : f32
      }
    %10 = linalg.fill ins(%cst_1 : f32) outs(%8 : tensor<f32>) -> tensor<f32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%reduced, %10 : tensor<f32>, tensor<f32>) outs(%8 : tensor<f32>) -> tensor<f32>
    return %11 : tensor<f32>
  }
}
