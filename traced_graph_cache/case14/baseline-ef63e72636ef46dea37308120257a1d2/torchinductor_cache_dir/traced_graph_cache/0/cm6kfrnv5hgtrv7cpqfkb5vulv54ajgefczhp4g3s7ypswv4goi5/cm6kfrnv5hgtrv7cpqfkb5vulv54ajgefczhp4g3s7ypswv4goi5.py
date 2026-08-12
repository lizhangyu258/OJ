
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class cm6kfrnv5hgtrv7cpqfkb5vulv54ajgefczhp4g3s7ypswv4goi5(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1):
        _to_copy = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float32);  arg0_1 = None
        _to_copy_1 = torch.ops.aten._to_copy.default(arg1_1, dtype = torch.float32);  arg1_1 = None
        mul = torch.ops.aten.mul.Scalar(_to_copy_1, 0.25);  _to_copy_1 = None
        add = torch.ops.aten.add.Tensor(_to_copy, mul);  mul = None
        _to_copy_2 = torch.ops.aten._to_copy.default(arg2_1, dtype = torch.float32);  arg2_1 = None
        expand = torch.ops.aten.expand.default(_to_copy_2, [arg3_1, arg4_1, arg5_1]);  _to_copy_2 = None
        add_1 = torch.ops.aten.add.Tensor(add, expand);  add = expand = None
        _to_copy_3 = torch.ops.aten._to_copy.default(arg6_1, dtype = torch.float32);  arg6_1 = None
        sigmoid = torch.ops.aten.sigmoid.default(_to_copy_3);  _to_copy_3 = None
        expand_1 = torch.ops.aten.expand.default(sigmoid, [arg3_1, arg4_1, arg5_1]);  sigmoid = arg3_1 = arg4_1 = arg5_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(add_1, expand_1);  add_1 = expand_1 = None
        relu = torch.ops.aten.relu.default(mul_1);  mul_1 = None
        mul_2 = torch.ops.aten.mul.Scalar(_to_copy, 0.125);  _to_copy = None
        add_2 = torch.ops.aten.add.Tensor(relu, mul_2);  relu = mul_2 = None
        return (add_2,)
        
