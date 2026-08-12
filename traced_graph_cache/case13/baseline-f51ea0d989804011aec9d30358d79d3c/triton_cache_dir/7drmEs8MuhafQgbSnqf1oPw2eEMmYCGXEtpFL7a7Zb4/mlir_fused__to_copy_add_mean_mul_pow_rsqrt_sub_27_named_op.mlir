
module {
  func.func @mlir_fused__to_copy_add_mean_mul_pow_rsqrt_sub_27(%arg0: tensor<?x?x?xf16>, %arg1: tensor<?x?x?xf16>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: tensor<?xf16>, %arg6: tensor<?xf16>) -> tensor<?x?x?xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>} {
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-02 : f64
    %cst_1 = arith.constant 1.000000e-04 : f64
    %cst_2 = arith.constant 2.500000e-01 : f32
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
    %4 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%3, %4 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %6 = tensor.empty(%dim, %dim_3, %dim_4) : tensor<?x?x?xi64>
    %7 = linalg.fill ins(%c1_i64 : i64) outs(%6 : tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
    %8 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%7 : tensor<?x?x?xi64>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%5, %8 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %9 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %11 = tensor.empty(%dim, %dim_3) : tensor<?x?xf32>
    %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %reduced = linalg.reduce ins(%10 : tensor<?x?x?xf32>) outs(%12 : tensor<?x?xf32>) dimensions = [2] 
      (%in: f32, %init: f32) {
        %50 = arith.addf %in, %init : f32
        linalg.yield %50 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0], [1, 2]] output_shape [%dim, %dim_3, 1] : tensor<?x?xf32> into tensor<?x?x1xf32>
    %13 = tensor.empty(%dim, %dim_3) : tensor<?x?x1xi64>
    %14 = linalg.fill ins(%arg2 : i64) outs(%13 : tensor<?x?x1xi64>) -> tensor<?x?x1xi64>
    %15 = tensor.empty(%dim, %dim_3) : tensor<?x?x1xf32>
    %16 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%14 : tensor<?x?x1xi64>) outs(%15 : tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded, %16 : tensor<?x?x1xf32>, tensor<?x?x1xf32>) outs(%15 : tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %18 = arith.index_cast %arg2 : i64 to index
    %19 = tensor.empty(%dim, %dim_3, %18) : tensor<?x?x?xf32>
    %collapsed = tensor.collapse_shape %17 [[0], [1, 2]] : tensor<?x?x1xf32> into tensor<?x?xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<?x?xf32>) outs(%19 : tensor<?x?x?xf32>) dimensions = [2] 
    %20 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted, %8 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %21 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%10, %20 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %22 = linalg.fill ins(%c2_i64 : i64) outs(%6 : tensor<?x?x?xi64>) -> tensor<?x?x?xi64>
    %23 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%22 : tensor<?x?x?xi64>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %24 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%21, %23 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %reduced_8 = linalg.reduce ins(%24 : tensor<?x?x?xf32>) outs(%12 : tensor<?x?xf32>) dimensions = [2] 
      (%in: f32, %init: f32) {
        %50 = arith.addf %in, %init : f32
        linalg.yield %50 : f32
      }
    %expanded_9 = tensor.expand_shape %reduced_8 [[0], [1, 2]] output_shape [%dim, %dim_3, 1] : tensor<?x?xf32> into tensor<?x?x1xf32>
    %25 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%expanded_9, %16 : tensor<?x?x1xf32>, tensor<?x?x1xf32>) outs(%15 : tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %26 = arith.truncf %cst_1 : f64 to f32
    %27 = linalg.fill ins(%26 : f32) outs(%15 : tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %28 = linalg.fill ins(%c1_i64 : i64) outs(%13 : tensor<?x?x1xi64>) -> tensor<?x?x1xi64>
    %29 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%28 : tensor<?x?x1xi64>) outs(%15 : tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %30 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%27, %29 : tensor<?x?x1xf32>, tensor<?x?x1xf32>) outs(%15 : tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %31 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%25, %30 : tensor<?x?x1xf32>, tensor<?x?x1xf32>) outs(%15 : tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %32 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%31 : tensor<?x?x1xf32>) outs(%15 : tensor<?x?x1xf32>) -> tensor<?x?x1xf32>
    %collapsed_10 = tensor.collapse_shape %32 [[0], [1, 2]] : tensor<?x?x1xf32> into tensor<?x?xf32>
    %broadcasted_11 = linalg.broadcast ins(%collapsed_10 : tensor<?x?xf32>) outs(%19 : tensor<?x?x?xf32>) dimensions = [2] 
    %33 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%21, %broadcasted_11 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dim_12 = tensor.dim %arg5, %c0 : tensor<?xf16>
    %34 = tensor.empty(%dim_12) : tensor<?xf32>
    %35 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg5 : tensor<?xf16>) outs(%34 : tensor<?xf32>) -> tensor<?xf32>
    %36 = arith.index_cast %arg3 : i64 to index
    %37 = arith.index_cast %arg4 : i64 to index
    %38 = tensor.empty(%36, %37, %dim_12) : tensor<?x?x?xf32>
    %broadcasted_13 = linalg.broadcast ins(%35 : tensor<?xf32>) outs(%38 : tensor<?x?x?xf32>) dimensions = [0, 1] 
    %39 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%33, %broadcasted_13 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %dim_14 = tensor.dim %arg6, %c0 : tensor<?xf16>
    %40 = tensor.empty(%dim_14) : tensor<?xf32>
    %41 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg6 : tensor<?xf16>) outs(%40 : tensor<?xf32>) -> tensor<?xf32>
    %42 = tensor.empty(%36, %37, %dim_14) : tensor<?x?x?xf32>
    %broadcasted_15 = linalg.broadcast ins(%41 : tensor<?xf32>) outs(%42 : tensor<?x?x?xf32>) dimensions = [0, 1] 
    %43 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%broadcasted_15, %8 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %44 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%39, %43 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %45 = arith.truncf %cst_0 : f64 to f32
    %46 = linalg.fill ins(%45 : f32) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %47 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%10, %46 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %48 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%47, %8 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %49 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%44, %48 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    return %49 : tensor<?x?x?xf32>
  }
}
