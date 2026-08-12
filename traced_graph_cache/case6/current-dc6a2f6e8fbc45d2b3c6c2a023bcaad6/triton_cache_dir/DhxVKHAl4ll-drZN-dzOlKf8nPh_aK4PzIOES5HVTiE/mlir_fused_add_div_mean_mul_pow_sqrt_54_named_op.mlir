
module {
  func.func @mlir_fused_add_div_mean_mul_pow_sqrt_54(%arg0: tensor<32x512xf32>, %arg1: tensor<512xf32>) -> (tensor<32x1xf32>, tensor<32x512xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 9.9999999999999995E-7 : f64
    %cst_1 = arith.constant 5.120000e+02 : f32
    %0 = tensor.empty() : tensor<32x512xi64>
    %1 = linalg.fill ins(%c2_i64 : i64) outs(%0 : tensor<32x512xi64>) -> tensor<32x512xi64>
    %2 = tensor.empty() : tensor<32x512xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%1 : tensor<32x512xi64>) outs(%2 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %3 : tensor<32x512xf32>, tensor<32x512xf32>) outs(%2 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %5 = tensor.empty() : tensor<32xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<32xf32>) -> tensor<32xf32>
    %reduced = linalg.reduce ins(%4 : tensor<32x512xf32>) outs(%6 : tensor<32xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %20 = arith.addf %in, %init : f32
        linalg.yield %20 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
    %7 = tensor.empty() : tensor<32x1xf32>
    %8 = linalg.fill ins(%cst_1 : f32) outs(%7 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded, %8 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%7 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %10 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%9 : tensor<32x1xf32>) outs(%7 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %11 = arith.truncf %cst_0 : f64 to f32
    %12 = linalg.fill ins(%11 : f32) outs(%7 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %13 = tensor.empty() : tensor<32x1xi64>
    %14 = linalg.fill ins(%c1_i64 : i64) outs(%13 : tensor<32x1xi64>) -> tensor<32x1xi64>
    %15 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%14 : tensor<32x1xi64>) outs(%7 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%12, %15 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%7 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%10, %16 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%7 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %collapsed = tensor.collapse_shape %17 [[0, 1]] : tensor<32x1xf32> into tensor<32xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<32xf32>) outs(%2 : tensor<32x512xf32>) dimensions = [1] 
    %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %broadcasted : tensor<32x512xf32>, tensor<32x512xf32>) outs(%2 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %broadcasted_2 = linalg.broadcast ins(%arg1 : tensor<512xf32>) outs(%2 : tensor<32x512xf32>) dimensions = [0] 
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%18, %broadcasted_2 : tensor<32x512xf32>, tensor<32x512xf32>) outs(%2 : tensor<32x512xf32>) -> tensor<32x512xf32>
    return %17, %19 : tensor<32x1xf32>, tensor<32x512xf32>
  }
}
