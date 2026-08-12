
module {
  func.func @mlir_fused__to_copy_add_mean_mul_pow_relu_sigmoid_sub_32(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x1x?xf32>, %arg2: i64, %arg3: i64, %arg4: i64) -> tensor<?x?x?xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg1, %c0 : tensor<?x1x?xf32>
    %dim_0 = tensor.dim %arg1, %c2 : tensor<?x1x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x1x?xi64>
    %1 = linalg.fill ins(%arg2 : i64) outs(%0 : tensor<?x1x?xi64>) -> tensor<?x1x?xi64>
    %2 = tensor.empty(%dim, %dim_0) : tensor<?x1x?xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%1 : tensor<?x1x?xi64>) outs(%2 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg1, %3 : tensor<?x1x?xf32>, tensor<?x1x?xf32>) outs(%2 : tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %5 = arith.index_cast %arg2 : i64 to index
    %6 = tensor.empty(%dim, %5, %dim_0) : tensor<?x?x?xf32>
    %collapsed = tensor.collapse_shape %4 [[0, 1], [2]] : tensor<?x1x?xf32> into tensor<?x?xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<?x?xf32>) outs(%6 : tensor<?x?x?xf32>) dimensions = [1] 
    %dim_1 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
    %dim_2 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
    %dim_3 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
    %7 = tensor.empty(%dim_1, %dim_2, %dim_3) : tensor<?x?x?xi64>
    %8 = linalg.fill ins(%c1_i64 : i64) outs(%7 : tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
    %9 = tensor.empty(%dim_1, %dim_2, %dim_3) : tensor<?x?x?xf32>
    %10 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%8 : tensor<?x?x?xi64>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %10 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %11 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %13 = linalg.fill ins(%c2_i64 : i64) outs(%7 : tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
    %14 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%13 : tensor<?x?x?xi64>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %15 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%12, %14 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%9 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    return %15 : tensor<?x?x?xf32>
  }
}
