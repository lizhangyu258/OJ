
module {
  func.func @mlir_fused__to_copy_add_mean_mul_relu_sub_18(%arg0: tensor<?x?xf16>, %arg1: tensor<1x?xf16>, %arg2: i64, %arg3: i64, %arg4: tensor<?x1xf16>) -> tensor<?x?xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %c1_i64 = arith.constant 1 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.250000e-01 : f32
    %cst_1 = arith.constant 7.500000e-01 : f32
    %cst_2 = arith.constant 2.500000e-01 : f32
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %dim_3 = tensor.dim %arg0, %c1 : tensor<?x?xf16>
    %0 = tensor.empty(%dim, %dim_3) : tensor<?x?xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<?x?xf16>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %dim_4 = tensor.dim %arg1, %c1 : tensor<1x?xf16>
    %2 = tensor.empty(%dim_4) : tensor<1x?xf32>
    %3 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<1x?xf16>) outs(%2 : tensor<1x?xf32>) -> tensor<1x?xf32>
    %4 = arith.index_cast %arg2 : i64 to index
    %5 = tensor.empty(%4, %dim_4) : tensor<?x?xf32>
    %collapsed = tensor.collapse_shape %3 [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<?xf32>) outs(%5 : tensor<?x?xf32>) dimensions = [0] 
    %6 = tensor.empty(%dim, %dim_3) : tensor<?x?xi64>
    %7 = linalg.fill ins(%c1_i64 : i64) outs(%6 : tensor<?x?xi64>) -> tensor<?x?xi64>
    %8 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%7 : tensor<?x?xi64>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %9 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %dim_5 = tensor.dim %arg4, %c0 : tensor<?x1xf16>
    %11 = tensor.empty(%dim_5) : tensor<?x1xf32>
    %12 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg4 : tensor<?x1xf16>) outs(%11 : tensor<?x1xf32>) -> tensor<?x1xf32>
    %13 = arith.index_cast %arg3 : i64 to index
    %14 = tensor.empty(%dim_5, %13) : tensor<?x?xf32>
    %collapsed_6 = tensor.collapse_shape %12 [[0, 1]] : tensor<?x1xf32> into tensor<?xf32>
    %broadcasted_7 = linalg.broadcast ins(%collapsed_6 : tensor<?xf32>) outs(%14 : tensor<?x?xf32>) dimensions = [1] 
    %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%10, %broadcasted_7 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %16 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} ins(%15 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %17 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %17 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%18, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%16, %19 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %21 = tensor.empty(%dim) : tensor<?xf32>
    %22 = linalg.fill ins(%cst : f32) outs(%21 : tensor<?xf32>) -> tensor<?xf32>
    %reduced = linalg.reduce ins(%20 : tensor<?x?xf32>) outs(%22 : tensor<?xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %37 = arith.addf %in, %init : f32
        linalg.yield %37 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [%dim, 1] : tensor<?xf32> into tensor<?x1xf32>
    %23 = tensor.empty(%dim) : tensor<?x1xi64>
    %24 = linalg.fill ins(%arg3 : i64) outs(%23 : tensor<?x1xi64>) -> tensor<?x1xi64>
    %25 = tensor.empty(%dim) : tensor<?x1xf32>
    %26 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%24 : tensor<?x1xi64>) outs(%25 : tensor<?x1xf32>) -> tensor<?x1xf32>
    %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded, %26 : tensor<?x1xf32>, tensor<?x1xf32>) outs(%25 : tensor<?x1xf32>) -> tensor<?x1xf32>
    %28 = tensor.empty(%dim, %13) : tensor<?x?xf32>
    %collapsed_8 = tensor.collapse_shape %27 [[0, 1]] : tensor<?x1xf32> into tensor<?xf32>
    %broadcasted_9 = linalg.broadcast ins(%collapsed_8 : tensor<?xf32>) outs(%28 : tensor<?x?xf32>) dimensions = [1] 
    %29 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_9, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %30 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%20, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %31 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %32 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%30, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %33 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %34 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%20, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %35 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%34, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %36 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%32, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %36 : tensor<?x?xf32>
  }
}
