
module {
  func.func @mlir_fused__safe_softmax_add_8(%arg0: tensor<32x128x128xf32>, %arg1: tensor<1x1x128x128xf16>) -> tensor<4x8x128x128xi1> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
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
