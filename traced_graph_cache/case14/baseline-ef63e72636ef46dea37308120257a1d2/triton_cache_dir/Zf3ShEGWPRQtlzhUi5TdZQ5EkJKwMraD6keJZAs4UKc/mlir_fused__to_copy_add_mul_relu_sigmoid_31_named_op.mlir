
module {
  func.func @mlir_fused__to_copy_add_mul_relu_sigmoid_31(%arg0: tensor<?x?x?xf16>, %arg1: tensor<?x?x?xf16>, %arg2: tensor<1x1x?xf16>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: tensor<?x?x1xf16>) -> tensor<?x?x?xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %c1_i64 = arith.constant 1 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 2.500000e-01 : f32
    %cst_2 = arith.constant 1.250000e-01 : f32
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf16>
    %dim_3 = tensor.dim %arg0, %c1 : tensor<?x?x?xf16>
    %dim_4 = tensor.dim %arg0, %c2 : tensor<?x?x?xf16>
    %0 = tensor.empty(%dim, %dim_3, %dim_4) : tensor<?x?x?xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<?x?x?xf16>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dim_5 = tensor.dim %arg1, %c0 : tensor<?x?x?xf16>
    %dim_6 = tensor.dim %arg1, %c1 : tensor<?x?x?xf16>
    %dim_7 = tensor.dim %arg1, %c2 : tensor<?x?x?xf16>
    %2 = tensor.empty(%dim_5, %dim_6, %dim_7) : tensor<?x?x?xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<?x?x?xf16>) outs(%2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %4 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%3, %4 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %6 = tensor.empty(%dim, %dim_3, %dim_4) : tensor<?x?x?xi64>
    %7 = linalg.fill ins(%c1_i64 : i64) outs(%6 : tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
    %8 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%7 : tensor<?x?x?xi64>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%5, %8 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %9 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dim_8 = tensor.dim %arg2, %c2 : tensor<1x1x?xf16>
    %11 = tensor.empty(%dim_8) : tensor<1x1x?xf32>
    %12 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg2 : tensor<1x1x?xf16>) outs(%11 : tensor<1x1x?xf32>) -> tensor<1x1x?xf32>
    %13 = arith.index_cast %arg3 : i64 to index
    %14 = arith.index_cast %arg4 : i64 to index
    %15 = tensor.empty(%13, %14, %dim_8) : tensor<?x?x?xf32>
    %collapsed = tensor.collapse_shape %12 [[0, 1, 2]] : tensor<1x1x?xf32> into tensor<?xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<?xf32>) outs(%15 : tensor<?x?x?xf32>) dimensions = [0, 1] 
    %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %8 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%10, %16 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dim_9 = tensor.dim %arg6, %c0 : tensor<?x?x1xf16>
    %dim_10 = tensor.dim %arg6, %c1 : tensor<?x?x1xf16>
    %18 = tensor.empty(%dim_9, %dim_10) : tensor<?x?x1xf32>
    %19 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg6 : tensor<?x?x1xf16>) outs(%18 : tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %20 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%19 : tensor<?x?x1xf32>) outs(%18 : tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %21 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%20 : tensor<?x?x1xf32>) outs(%18 : tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%21, %cst : tensor<?x?x1xf32>, f32) outs(%18 : tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %22 : f32, tensor<?x?x1xf32>) outs(%18 : tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %24 = arith.index_cast %arg5 : i64 to index
    %25 = tensor.empty(%dim_9, %dim_10, %24) : tensor<?x?x?xf32>
    %collapsed_11 = tensor.collapse_shape %23 [[0], [1, 2]] : tensor<?x?x1xf32> into tensor<?x?xf32>
    %broadcasted_12 = linalg.broadcast ins(%collapsed_11 : tensor<?x?xf32>) outs(%25 : tensor<?x?x?xf32>) dimensions = [2] 
    %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%17, %broadcasted_12 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %27 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} ins(%26 : tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %28 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %29 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %28 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %30 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%29, %8 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %31 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%27, %30 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    return %31 : tensor<?x?x?xf32>
  }
}
