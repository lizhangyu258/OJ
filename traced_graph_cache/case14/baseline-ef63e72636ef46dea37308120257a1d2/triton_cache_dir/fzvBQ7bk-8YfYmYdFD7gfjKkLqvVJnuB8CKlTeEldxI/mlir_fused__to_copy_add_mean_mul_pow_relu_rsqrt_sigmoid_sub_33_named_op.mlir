
module {
  func.func @mlir_fused__to_copy_add_mean_mul_pow_relu_rsqrt_sigmoid_sub_33(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x1x?xf32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: tensor<?x1x?xf32>) -> tensor<?x?x?xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %c1_i64 = arith.constant 1 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 1.000000e-04 : f64
    %cst_0 = arith.constant 7.500000e-01 : f32
    %cst_1 = arith.constant 2.500000e-01 : f32
    %dim = tensor.dim %arg1, %c0 : tensor<?x1x?xf32>
    %dim_2 = tensor.dim %arg1, %c2 : tensor<?x1x?xf32>
    %0 = tensor.empty(%dim, %dim_2) : tensor<?x1x?xi64>
    %1 = linalg.fill ins(%arg2 : i64) outs(%0 : tensor<?x1x?xi64>) -> tensor<?x1x?xi64>
    %2 = tensor.empty(%dim, %dim_2) : tensor<?x1x?xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%1 : tensor<?x1x?xi64>) outs(%2 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg1, %3 : tensor<?x1x?xf32>, tensor<?x1x?xf32>) outs(%2 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %5 = arith.index_cast %arg2 : i64 to index
    %6 = tensor.empty(%dim, %5, %dim_2) : tensor<?x?x?xf32>
    %collapsed = tensor.collapse_shape %4 [[0, 1], [2]] : tensor<?x1x?xf32> into tensor<?x?xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<?x?xf32>) outs(%6 : tensor<?x?x?xf32>) dimensions = [1] 
    %dim_3 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
    %dim_4 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
    %dim_5 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
    %7 = tensor.empty(%dim_3, %dim_4, %dim_5) : tensor<?x?x?xi64>
    %8 = linalg.fill ins(%c1_i64 : i64) outs(%7 : tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
    %9 = tensor.empty(%dim_3, %dim_4, %dim_5) : tensor<?x?x?xf32>
    %10 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%8 : tensor<?x?x?xi64>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %10 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %11 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dim_6 = tensor.dim %arg5, %c0 : tensor<?x1x?xf32>
    %dim_7 = tensor.dim %arg5, %c2 : tensor<?x1x?xf32>
    %13 = tensor.empty(%dim_6, %dim_7) : tensor<?x1x?xi64>
    %14 = linalg.fill ins(%arg2 : i64) outs(%13 : tensor<?x1x?xi64>) -> tensor<?x1x?xi64>
    %15 = tensor.empty(%dim_6, %dim_7) : tensor<?x1x?xf32>
    %16 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%14 : tensor<?x1x?xi64>) outs(%15 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg5, %16 : tensor<?x1x?xf32>, tensor<?x1x?xf32>) outs(%15 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %18 = arith.truncf %cst : f64 to f32
    %19 = linalg.fill ins(%18 : f32) outs(%15 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %20 = linalg.fill ins(%c1_i64 : i64) outs(%13 : tensor<?x1x?xi64>) -> tensor<?x1x?xi64>
    %21 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%20 : tensor<?x1x?xi64>) outs(%15 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%19, %21 : tensor<?x1x?xf32>, tensor<?x1x?xf32>) outs(%15 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%17, %22 : tensor<?x1x?xf32>, tensor<?x1x?xf32>) outs(%15 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %24 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%23 : tensor<?x1x?xf32>) outs(%15 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %25 = tensor.empty(%dim_6, %5, %dim_7) : tensor<?x?x?xf32>
    %collapsed_8 = tensor.collapse_shape %24 [[0, 1], [2]] : tensor<?x1x?xf32> into tensor<?x?xf32>
    %broadcasted_9 = linalg.broadcast ins(%collapsed_8 : tensor<?x?xf32>) outs(%25 : tensor<?x?x?xf32>) dimensions = [1] 
    %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%12, %broadcasted_9 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %27 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%26, %27 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %29 = linalg.fill ins(%cst_1 : f32) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %30 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %29 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %31 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%30, %10 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %32 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%28, %31 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    return %32 : tensor<?x?x?xf32>
  }
}
