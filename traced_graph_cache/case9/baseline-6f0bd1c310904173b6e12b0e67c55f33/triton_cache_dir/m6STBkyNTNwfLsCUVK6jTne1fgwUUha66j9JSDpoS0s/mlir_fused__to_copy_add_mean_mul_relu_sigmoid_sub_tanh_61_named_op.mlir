
module {
  func.func @mlir_fused__to_copy_add_mean_mul_relu_sigmoid_sub_tanh_61(%arg0: tensor<256x2048xf16>, %arg1: tensor<1x2048xf16>, %arg2: tensor<256x2048xf16>) -> tensor<256x2048xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 2.000000e-01 : f64
    %cst_2 = arith.constant 5.000000e-01 : f32
    %cst_3 = arith.constant 2.500000e-01 : f32
    %cst_4 = arith.constant 1.250000e-01 : f32
    %cst_5 = arith.constant 2.048000e+03 : f32
    %0 = tensor.empty() : tensor<256x2048xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %2 = tensor.empty() : tensor<1x2048xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<1x2048xf16>) outs(%2 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %collapsed = tensor.collapse_shape %3 [[0, 1]] : tensor<1x2048xf32> into tensor<2048xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<2048xf32>) outs(%0 : tensor<256x2048xf32>) dimensions = [0] 
    %4 = tensor.empty() : tensor<256x2048xi64>
    %5 = linalg.fill ins(%c1_i64 : i64) outs(%4 : tensor<256x2048xi64>) -> tensor<256x2048xi64>
    %6 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%5 : tensor<256x2048xi64>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %6 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %7 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %9 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%8 : tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %10 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%9 : tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%10, %cst : tensor<256x2048xf32>, f32) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %11 : f32, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %13 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg2 : tensor<256x2048xf16>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%13, %7 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %15 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tanh>} ins(%14 : tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%12, %15 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%13, %6 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %17 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%18, %7 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %20 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} ins(%19 : tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %21 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%20, %21 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%22, %6 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %24 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%16, %23 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %25 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%12, %20 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %26 = linalg.fill ins(%cst_3 : f32) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%25, %26 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%27, %6 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %29 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%24, %28 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %30 = linalg.fill ins(%cst_4 : f32) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %31 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %30 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %32 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%31, %6 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %33 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%29, %32 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %34 = tensor.empty() : tensor<256xf32>
    %35 = linalg.fill ins(%cst_0 : f32) outs(%34 : tensor<256xf32>) -> tensor<256xf32>
    %reduced = linalg.reduce ins(%33 : tensor<256x2048xf32>) outs(%35 : tensor<256xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %44 = arith.addf %in, %init : f32
        linalg.yield %44 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [256, 1] : tensor<256xf32> into tensor<256x1xf32>
    %36 = tensor.empty() : tensor<256x1xf32>
    %37 = linalg.fill ins(%cst_5 : f32) outs(%36 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %38 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded, %37 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%36 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %39 = arith.truncf %cst_1 : f64 to f32
    %40 = linalg.fill ins(%39 : f32) outs(%36 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %41 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%38, %40 : tensor<256x1xf32>, tensor<256x1xf32>) outs(%36 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %collapsed_6 = tensor.collapse_shape %41 [[0, 1]] : tensor<256x1xf32> into tensor<256xf32>
    %broadcasted_7 = linalg.broadcast ins(%collapsed_6 : tensor<256xf32>) outs(%0 : tensor<256x2048xf32>) dimensions = [1] 
    %42 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_7, %6 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    %43 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%33, %42 : tensor<256x2048xf32>, tensor<256x2048xf32>) outs(%0 : tensor<256x2048xf32>) -> tensor<256x2048xf32>
    return %43 : tensor<256x2048xf32>
  }
}
