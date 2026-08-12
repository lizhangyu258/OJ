
module {
  func.func @mlir_fused__to_copy_add_mean_mul_pow_rsqrt_sub_39(%arg0: tensor<256x2048xf16>, %arg1: tensor<256x2048xf16>, %arg2: tensor<1x2048xf16>) -> tensor<256x2048xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-02 : f64
    %cst_1 = arith.constant 1.000000e-03 : f64
    %cst_2 = arith.constant 2.500000e-01 : f32
    %cst_3 = arith.constant 2.048000e+03 : f32
    %0 = tensor.empty() : tensor<256x2048xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %2 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %4 = tensor.empty() : tensor<256x2048xi64>
    %5 = linalg.fill ins(%c1_i64 : i64) outs(%4 : tensor<256x2048xi64>) -> tensor<256x2048xi64>
    %6 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%5 : tensor<256x2048xi64>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %6 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %7 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %9 = tensor.empty() : tensor<1x2048xf32>
    %10 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg2 : tensor<1x2048xf16>) outs(%9 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %collapsed = tensor.collapse_shape %10 [[0, 1]] : tensor<1x2048xf32> into tensor<2048xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<2048xf32>) outs(%0 : tensor<256x2048xf32>) dimensions = [0] 
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%8, %broadcasted : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %12 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %12 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%13, %6 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%11, %14 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %16 = tensor.empty() : tensor<256xf32>
    %17 = linalg.fill ins(%cst : f32) outs(%16 : tensor<256xf32>) -> tensor<256xf32>
    %reduced = linalg.reduce ins(%15 : tensor<256x2048xf32>) outs(%17 : tensor<256xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %41 = arith.addf %in, %init : f32
        linalg.yield %41 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [256, 1] : tensor<256xf32> into tensor<256x1xf32>
    %18 = tensor.empty() : tensor<256x1xf32>
    %19 = linalg.fill ins(%cst_3 : f32) outs(%18 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded, %19 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%18 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %collapsed_4 = tensor.collapse_shape %20 [[0, 1]] : tensor<256x1xf32> into tensor<256xf32>
    %broadcasted_5 = linalg.broadcast ins(%collapsed_4 : tensor<256xf32>) outs(%0 : tensor<256x2048xf32>) dimensions = [1] 
    %21 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_5, %6 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%15, %21 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %23 = linalg.fill ins(%c2_i64 : i64) outs(%4 : tensor<256x2048xi64>) -> tensor<256x2048xi64>
    %24 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%23 : tensor<256x2048xi64>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %25 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%22, %24 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %reduced_6 = linalg.reduce ins(%25 : tensor<256x2048xf32>) outs(%17 : tensor<256xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %41 = arith.addf %in, %init : f32
        linalg.yield %41 : f32
      }
    %expanded_7 = tensor.expand_shape %reduced_6 [[0, 1]] output_shape [256, 1] : tensor<256xf32> into tensor<256x1xf32>
    %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded_7, %19 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%18 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %27 = arith.truncf %cst_1 : f64 to f32
    %28 = linalg.fill ins(%27 : f32) outs(%18 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %29 = tensor.empty() : tensor<256x1xi64>
    %30 = linalg.fill ins(%c1_i64 : i64) outs(%29 : tensor<256x1xi64>) -> tensor<256x1xi64>
    %31 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%30 : tensor<256x1xi64>) outs(%18 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %32 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%28, %31 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%18 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %33 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%26, %32 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%18 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %34 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%33 : tensor<256x1xf32>) outs(%18 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %collapsed_8 = tensor.collapse_shape %34 [[0, 1]] : tensor<256x1xf32> into tensor<256xf32>
    %broadcasted_9 = linalg.broadcast ins(%collapsed_8 : tensor<256xf32>) outs(%0 : tensor<256x2048xf32>) dimensions = [1] 
    %35 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%22, %broadcasted_9 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %36 = arith.truncf %cst_0 : f64 to f32
    %37 = linalg.fill ins(%36 : f32) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %38 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%15, %37 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %39 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%38, %6 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %40 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%35, %39 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    return %40 : tensor<256x2048xf32>
  }
}
