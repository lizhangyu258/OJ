
module {
  func.func @mlir_fused_add_mean_mul_sub_sum_0(%arg0: tensor<16x1000xf32>, %arg1: tensor<16x1000xf32>) -> (tensor<16xf32>, tensor<16xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e-01 : f64
    %cst_1 = arith.constant 5.000000e-01 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant 1.500000e+00 : f32
    %0 = tensor.empty() : tensor<16x1000xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %2 = tensor.empty() : tensor<16x1000xi64>
    %3 = linalg.fill ins(%c1_i64 : i64) outs(%2 : tensor<16x1000xi64>) -> tensor<16x1000xi64>
    %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%3 : tensor<16x1000xi64>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %4 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %5 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%6, %4 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %7 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %9 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg0 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %10 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %10 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%11, %4 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%9, %12 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %14 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%14, %4 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%13, %15 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%8, %16 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %18 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%16, %4 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %19 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%8, %18 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %20 = arith.truncf %cst_0 : f64 to f32
    %21 = linalg.fill ins(%20 : f32) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %22 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%19, %21 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %23 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%22, %4 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %24 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%17, %23 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %25 = linalg.fill ins(%cst_3 : f32) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %26 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg1, %25 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%26, %4 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %28 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%24, %27 : tensor<16x1000xf32>, tensor<16x1000xf32>) outs(%0 : tensor<16x1000xf32>) -> tensor<16x1000xf32>
    %29 = tensor.empty() : tensor<16xf32>
    %30 = linalg.fill ins(%cst : f32) outs(%29 : tensor<16xf32>) -> tensor<16xf32>
    %reduced = linalg.reduce ins(%28 : tensor<16x1000xf32>) outs(%30 : tensor<16xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %31 = arith.addf %in, %init : f32
        linalg.yield %31 : f32
      }
    return %reduced, %reduced : tensor<16xf32>, tensor<16xf32>
  }
}
