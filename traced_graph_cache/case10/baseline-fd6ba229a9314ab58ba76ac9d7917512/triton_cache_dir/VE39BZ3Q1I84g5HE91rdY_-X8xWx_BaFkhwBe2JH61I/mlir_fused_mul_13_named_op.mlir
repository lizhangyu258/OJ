
module {
  func.func @mlir_fused_mul_13(%arg0: tensor<4x8x128x64xf16>) -> tensor<4x8x64x128xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.35355339059327379 : f64
    %0 = tensor.empty() : tensor<4x8x128x64xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x8x128x64xf16>) outs(%0 : tensor<4x8x128x64xf32>) -> tensor<4x8x128x64xf32>
    %2 = tensor.empty() : tensor<4x8x64x128xf32>
    %transposed = linalg.transpose ins(%1 : tensor<4x8x128x64xf32>) outs(%2 : tensor<4x8x64x128xf32>) permutation = [0, 1, 3, 2] 
    %3 = arith.truncf %cst : f64 to f32
    %4 = linalg.fill ins(%3 : f32) outs(%2 : tensor<4x8x64x128xf32>) -> tensor<4x8x64x128xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%transposed, %4 : tensor<4x8x64x128xf32>, tensor<4x8x64x128xf32>) outs(%2 : tensor<4x8x64x128xf32>) -> tensor<4x8x64x128xf32>
    return %5 : tensor<4x8x64x128xf32>
  }
}
