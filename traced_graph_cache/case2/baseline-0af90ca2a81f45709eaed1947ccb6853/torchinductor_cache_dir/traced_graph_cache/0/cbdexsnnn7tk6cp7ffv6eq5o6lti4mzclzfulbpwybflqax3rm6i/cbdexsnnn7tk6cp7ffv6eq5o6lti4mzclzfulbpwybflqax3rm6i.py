
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class cbdexsnnn7tk6cp7ffv6eq5o6lti4mzclzfulbpwybflqax3rm6i(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1):
        _to_copy = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float32);  arg0_1 = None
        _to_copy_1 = torch.ops.aten._to_copy.default(arg1_1, dtype = torch.float32);  arg1_1 = None
        mul = torch.ops.aten.mul.Tensor(_to_copy, _to_copy_1)
        add = torch.ops.aten.add.Tensor(mul, _to_copy);  mul = _to_copy = None
        _to_copy_2 = torch.ops.aten._to_copy.default(arg2_1, dtype = torch.float32);  arg2_1 = None
        expand = torch.ops.aten.expand.default(_to_copy_2, [256, 2048]);  _to_copy_2 = None
        mul_1 = torch.ops.aten.mul.Tensor(add, expand);  add = expand = None
        mul_2 = torch.ops.aten.mul.Scalar(_to_copy_1, 0.25);  _to_copy_1 = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(add_1, [1], True)
        div = torch.ops.aten.div.Scalar(sum_1, 2048.0);  sum_1 = None
        expand_1 = torch.ops.aten.expand.default(div, [256, 2048]);  div = None
        sub = torch.ops.aten.sub.Tensor(add_1, expand_1);  expand_1 = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(sub, 2)
        sum_2 = torch.ops.aten.sum.dim_IntList(pow_1, [1], True);  pow_1 = None
        div_1 = torch.ops.aten.div.Scalar(sum_2, 2048.0);  sum_2 = None
        add_2 = torch.ops.aten.add.Scalar(div_1, 0.001);  div_1 = None
        rsqrt = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        expand_2 = torch.ops.aten.expand.default(rsqrt, [256, 2048]);  rsqrt = None
        mul_3 = torch.ops.aten.mul.Tensor(sub, expand_2);  sub = expand_2 = None
        mul_4 = torch.ops.aten.mul.Scalar(add_1, 0.05);  add_1 = None
        add_3 = torch.ops.aten.add.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
        return (add_3,)
        
