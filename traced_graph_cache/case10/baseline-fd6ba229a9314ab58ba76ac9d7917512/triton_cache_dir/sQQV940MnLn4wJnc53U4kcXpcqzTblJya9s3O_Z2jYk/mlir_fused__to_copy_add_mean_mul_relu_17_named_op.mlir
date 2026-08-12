
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
