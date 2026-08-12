
module {
  func.func @mlir_fused__to_copy_add_mean_mul_relu_43(%arg0: tensor<128x512xf32>, %arg1: tensor<128x1xf16>, %arg2: tensor<128x512xf16>) -> tensor<128x512xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
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
