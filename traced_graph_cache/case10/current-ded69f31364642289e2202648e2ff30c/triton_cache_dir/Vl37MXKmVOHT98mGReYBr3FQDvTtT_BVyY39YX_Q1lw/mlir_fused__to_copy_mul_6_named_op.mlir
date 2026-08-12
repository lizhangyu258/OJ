
module {
  func.func @mlir_fused__to_copy_mul_6(%arg0: tensor<4x8x128x64xf16>) -> tensor<4x8x128x64xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %cst = arith.constant 0.35355339059327379 : f64
    %0 = tensor.empty() : tensor<4x8x128x64xf32>
    %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x8x128x64xf16>) outs(%0 : tensor<4x8x128x64xf32>) -> tensor<4x8x128x64xf32>
    %2 = arith.truncf %cst : f64 to f32
    %3 = linalg.fill ins(%2 : f32) outs(%0 : tensor<4x8x128x64xf32>) -> tensor<4x8x128x64xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %3 : tensor<4x8x128x64xf32>, tensor<4x8x128x64xf32>) outs(%0 : tensor<4x8x128x64xf32>) -> tensor<4x8x128x64xf32>
    return %4 : tensor<4x8x128x64xf32>
  }
}
