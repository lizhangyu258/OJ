
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class c7ox6xdfndkah2nkrmbzfcafehszl3lrc2ks7zlr6yqpkn2qoezc(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1):
        _to_copy = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float32);  arg0_1 = None
        _to_copy_1 = torch.ops.aten._to_copy.default(arg1_1, dtype = torch.float32);  arg1_1 = None
        mul = torch.ops.aten.mul.Scalar(_to_copy_1, 0.25);  _to_copy_1 = None
        add = torch.ops.aten.add.Tensor(_to_copy, mul);  _to_copy = mul = None
        sum_1 = torch.ops.aten.sum.dim_IntList(add, [2], True)
        div = torch.ops.aten.div.Scalar(sum_1, arg2_1);  sum_1 = None
        expand = torch.ops.aten.expand.default(div, [arg3_1, arg4_1, arg2_1]);  div = None
        sub = torch.ops.aten.sub.Tensor(add, expand);  expand = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(sub, 2)
        sum_2 = torch.ops.aten.sum.dim_IntList(pow_1, [2], True);  pow_1 = None
        div_1 = torch.ops.aten.div.Scalar(sum_2, arg2_1);  sum_2 = None
        add_1 = torch.ops.aten.add.Scalar(div_1, 0.0001);  div_1 = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        expand_1 = torch.ops.aten.expand.default(rsqrt, [arg3_1, arg4_1, arg2_1]);  rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(sub, expand_1);  sub = expand_1 = None
        _to_copy_2 = torch.ops.aten._to_copy.default(arg5_1, dtype = torch.float32);  arg5_1 = None
        expand_2 = torch.ops.aten.expand.default(_to_copy_2, [arg3_1, arg4_1, arg2_1]);  _to_copy_2 = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, expand_2);  mul_1 = expand_2 = None
        _to_copy_3 = torch.ops.aten._to_copy.default(arg6_1, dtype = torch.float32);  arg6_1 = None
        expand_3 = torch.ops.aten.expand.default(_to_copy_3, [arg3_1, arg4_1, arg2_1]);  _to_copy_3 = arg3_1 = arg4_1 = arg2_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_2, expand_3);  mul_2 = expand_3 = None
        mul_3 = torch.ops.aten.mul.Scalar(add, 0.05);  add = None
        add_3 = torch.ops.aten.add.Tensor(add_2, mul_3);  add_2 = mul_3 = None
        return (add_3,)
        
