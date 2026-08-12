
module {
  func.func @mlir_fused_add_mean_mul_pow_rsqrt_sub_51(%arg0: tensor<32x512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<32x512xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 9.9999999999999995E-7 : f64
    %cst_1 = arith.constant 5.120000e+02 : f32
    %0 = tensor.empty() : tensor<32xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
    %reduced = linalg.reduce ins(%arg0 : tensor<32x512xf32>) outs(%1 : tensor<32xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %27 = arith.addf %in, %init : f32
        linalg.yield %27 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
    %2 = tensor.empty() : tensor<32x1xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded, %3 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%2 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %5 = tensor.empty() : tensor<32x512xf32>
    %collapsed = tensor.collapse_shape %4 [[0, 1]] : tensor<32x1xf32> into tensor<32xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<32xf32>) outs(%5 : tensor<32x512xf32>) dimensions = [1] 
    %6 = tensor.empty() : tensor<32x512xi64>
    %7 = linalg.fill ins(%c1_i64 : i64) outs(%6 : tensor<32x512xi64>) -> tensor<32x512xi64>
    %8 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%7 : tensor<32x512xi64>) outs(%5 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %8 : tensor<32x512xf32>, tensor<32x512xf32>) outs(%5 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %9 : tensor<32x512xf32>, tensor<32x512xf32>) outs(%5 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %11 = linalg.fill ins(%c2_i64 : i64) outs(%6 : tensor<32x512xi64>) -> tensor<32x512xi64>
    %12 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%11 : tensor<32x512xi64>) outs(%5 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %13 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%10, %12 : tensor<32x512xf32>, tensor<32x512xf32>) outs(%5 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %reduced_2 = linalg.reduce ins(%13 : tensor<32x512xf32>) outs(%1 : tensor<32xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %27 = arith.addf %in, %init : f32
        linalg.yield %27 : f32
      }
    %expanded_3 = tensor.expand_shape %reduced_2 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
    %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded_3, %3 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%2 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %15 = arith.truncf %cst_0 : f64 to f32
    %16 = linalg.fill ins(%15 : f32) outs(%2 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %17 = tensor.empty() : tensor<32x1xi64>
    %18 = linalg.fill ins(%c1_i64 : i64) outs(%17 : tensor<32x1xi64>) -> tensor<32x1xi64>
    %19 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%18 : tensor<32x1xi64>) outs(%2 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%16, %19 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%2 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %21 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%14, %20 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%2 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %22 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%21 : tensor<32x1xf32>) outs(%2 : tensor<32x1xf32>) -> tensor<32x1xf32>
    %collapsed_4 = tensor.collapse_shape %22 [[0, 1]] : tensor<32x1xf32> into tensor<32xf32>
    %broadcasted_5 = linalg.broadcast ins(%collapsed_4 : tensor<32xf32>) outs(%5 : tensor<32x512xf32>) dimensions = [1] 
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%10, %broadcasted_5 : tensor<32x512xf32>, tensor<32x512xf32>) outs(%5 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %broadcasted_6 = linalg.broadcast ins(%arg1 : tensor<512xf32>) outs(%5 : tensor<32x512xf32>) dimensions = [0] 
    %24 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%23, %broadcasted_6 : tensor<32x512xf32>, tensor<32x512xf32>) outs(%5 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %broadcasted_7 = linalg.broadcast ins(%arg2 : tensor<512xf32>) outs(%5 : tensor<32x512xf32>) dimensions = [0] 
    %25 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_7, %8 : tensor<32x512xf32>, tensor<32x512xf32>) outs(%5 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%24, %25 : tensor<32x512xf32>, tensor<32x512xf32>) outs(%5 : tensor<32x512xf32>) -> tensor<32x512xf32>
    return %26 : tensor<32x512xf32>
  }
}
