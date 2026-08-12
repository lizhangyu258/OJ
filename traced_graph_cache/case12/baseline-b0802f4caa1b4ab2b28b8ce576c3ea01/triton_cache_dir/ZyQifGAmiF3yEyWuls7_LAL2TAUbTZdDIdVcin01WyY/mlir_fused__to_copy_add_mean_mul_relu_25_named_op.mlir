
module {
  func.func @mlir_fused__to_copy_add_mean_mul_relu_25(%arg0: tensor<?x?xf32>, %arg1: tensor<?x1xf16>, %arg2: i64, %arg3: i64) -> tensor<?x?xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %c1_i64 = arith.constant 1 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-02 : f64
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim, %dim_1) : tensor<?x?xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} ins(%arg0 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %dim_2 = tensor.dim %arg1, %c0 : tensor<?x1xf16>
    %2 = tensor.empty(%dim_2) : tensor<?x1xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<?x1xf16>) outs(%2 : tensor<?x1xf32>) -> tensor<?x1xf32>
    %4 = arith.index_cast %arg3 : i64 to index
    %5 = tensor.empty(%dim_2, %4) : tensor<?x?xf32>
    %collapsed = tensor.collapse_shape %3 [[0, 1]] : tensor<?x1xf32> into tensor<?xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<?xf32>) outs(%5 : tensor<?x?xf32>) dimensions = [1] 
    %6 = tensor.empty(%dim, %dim_1) : tensor<?x?xi64>
    %7 = linalg.fill ins(%c1_i64 : i64) outs(%6 : tensor<?x?xi64>) -> tensor<?x?xi64>
    %8 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%7 : tensor<?x?xi64>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %9 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %11 = tensor.empty(%dim) : tensor<?xf32>
    %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<?xf32>) -> tensor<?xf32>
    %reduced = linalg.reduce ins(%10 : tensor<?x?xf32>) outs(%12 : tensor<?xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %24 = arith.addf %in, %init : f32
        linalg.yield %24 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [%dim, 1] : tensor<?xf32> into tensor<?x1xf32>
    %13 = tensor.empty(%dim) : tensor<?x1xi64>
    %14 = linalg.fill ins(%arg3 : i64) outs(%13 : tensor<?x1xi64>) -> tensor<?x1xi64>
    %15 = tensor.empty(%dim) : tensor<?x1xf32>
    %16 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%14 : tensor<?x1xi64>) outs(%15 : tensor<?x1xf32>) -> tensor<?x1xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded, %16 : tensor<?x1xf32>, tensor<?x1xf32>) outs(%15 : tensor<?x1xf32>) -> tensor<?x1xf32>
    %18 = arith.truncf %cst_0 : f64 to f32
    %19 = linalg.fill ins(%18 : f32) outs(%15 : tensor<?x1xf32>) -> tensor<?x1xf32>
    %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%17, %19 : tensor<?x1xf32>, tensor<?x1xf32>) outs(%15 : tensor<?x1xf32>) -> tensor<?x1xf32>
    %21 = tensor.empty(%dim, %4) : tensor<?x?xf32>
    %collapsed_3 = tensor.collapse_shape %20 [[0, 1]] : tensor<?x1xf32> into tensor<?xf32>
    %broadcasted_4 = linalg.broadcast ins(%collapsed_3 : tensor<?xf32>) outs(%21 : tensor<?x?xf32>) dimensions = [1] 
    %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_4, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%10, %22 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %23 : tensor<?x?xf32>
  }
}
