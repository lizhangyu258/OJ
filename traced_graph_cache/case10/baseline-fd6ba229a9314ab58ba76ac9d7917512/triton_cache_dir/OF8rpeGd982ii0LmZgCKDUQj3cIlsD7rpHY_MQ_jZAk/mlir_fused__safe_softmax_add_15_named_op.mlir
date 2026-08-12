
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
