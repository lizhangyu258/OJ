
module {
  func.func @mlir_fused__to_copy_add_mean_mul_relu_49(%arg0: tensor<256x2048xf16>, %arg1: tensor<1x2048xf16>, %arg2: tensor<256x2048xf16>, %arg3: tensor<256x1xf16>) -> tensor<256x2048xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 2.500000e-01 : f32
    %cst_2 = arith.constant 2.048000e+03 : f32
    %cst_3 = arith.constant 1.250000e-01 : f32
    %0 = tensor.empty() : tensor<256x2048xf16>
    %collapsed = tensor.collapse_shape %arg1 [[0, 1]] : tensor<1x2048xf16> into tensor<2048xf16>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<2048xf16>) outs(%0 : tensor<256x2048xf16>) dimensions = [0] 
    %1 = tensor.empty() : tensor<256x2048xi64>
    %2 = linalg.fill ins(%c1_i64 : i64) outs(%1 : tensor<256x2048xi64>) -> tensor<256x2048xi64>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%2 : tensor<256x2048xi64>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %3 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %4 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%5, %arg2 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %collapsed_4 = tensor.collapse_shape %arg3 [[0, 1]] : tensor<256x1xf16> into tensor<256xf16>
    %broadcasted_5 = linalg.broadcast ins(%collapsed_4 : tensor<256xf16>) outs(%0 : tensor<256x2048xf16>) dimensions = [1] 
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_5, %3 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%6, %7 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %9 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} ins(%8 : tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %10 = tensor.empty() : tensor<256x2048xf32>
    %11 = linalg.fill ins(%cst_1 : f32) outs(%10 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %12 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%11 : tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%8, %12 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%13, %3 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%9, %14 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %16 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%15 : tensor<256x2048xf16>) outs(%10 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %17 = tensor.empty() : tensor<256xf32>
    %18 = linalg.fill ins(%cst_0 : f32) outs(%17 : tensor<256xf32>) -> tensor<256xf32>
    %reduced = linalg.reduce ins(%16 : tensor<256x2048xf32>) outs(%18 : tensor<256xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %30 = arith.addf %in, %init : f32
        linalg.yield %30 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [256, 1] : tensor<256xf32> into tensor<256x1xf32>
    %19 = tensor.empty() : tensor<256x1xf32>
    %20 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %21 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded, %20 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%19 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %22 = tensor.empty() : tensor<256x1xf16>
    %23 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%21 : tensor<256x1xf32>) outs(%22 : tensor<256x1xf16>) -> tensor<256x1xf16>
    %24 = linalg.fill ins(%cst_3 : f32) outs(%19 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %25 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%24 : tensor<256x1xf32>) outs(%22 : tensor<256x1xf16>) -> tensor<256x1xf16>
    %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%23, %25 : tensor<256x1xf16>, tensor<256x1xf16>) outs(%22 : tensor<256x1xf16>) -> tensor<256x1xf16>
    %collapsed_6 = tensor.collapse_shape %26 [[0, 1]] : tensor<256x1xf16> into tensor<256xf16>
    %broadcasted_7 = linalg.broadcast ins(%collapsed_6 : tensor<256xf16>) outs(%0 : tensor<256x2048xf16>) dimensions = [1] 
    %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_7, %3 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%15, %27 : tensor<256x2048xf16>, tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf16>) -> tensor<256x2048xf16>
    %29 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%28 : tensor<256x2048xf16>) outs(%10 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    return %29 : tensor<256x2048xf32>
  }
}
