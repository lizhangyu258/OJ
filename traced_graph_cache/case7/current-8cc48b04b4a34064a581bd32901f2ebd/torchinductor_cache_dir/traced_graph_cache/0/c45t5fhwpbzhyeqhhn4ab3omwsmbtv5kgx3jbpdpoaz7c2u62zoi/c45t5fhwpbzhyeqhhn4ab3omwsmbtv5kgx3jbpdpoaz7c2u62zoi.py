
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class c45t5fhwpbzhyeqhhn4ab3omwsmbtv5kgx3jbpdpoaz7c2u62zoi(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1):
        sum_1 = torch.ops.aten.sum.dim_IntList(arg0_1, [1], True)
        div = torch.ops.aten.div.Scalar(sum_1, 2048.0);  sum_1 = None
        expand = torch.ops.aten.expand.default(div, [256, 2048]);  div = None
        sub = torch.ops.aten.sub.Tensor(arg0_1, expand);  expand = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(sub, 2)
        sum_2 = torch.ops.aten.sum.dim_IntList(pow_1, [1], True);  pow_1 = None
        div_1 = torch.ops.aten.div.Scalar(sum_2, 2048.0);  sum_2 = None
        add = torch.ops.aten.add.Scalar(div_1, 0.0001);  div_1 = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        expand_1 = torch.ops.aten.expand.default(rsqrt, [256, 2048]);  rsqrt = None
        mul = torch.ops.aten.mul.Tensor(sub, expand_1);  sub = expand_1 = None
        mul_1 = torch.ops.aten.mul.Scalar(mul, 0.75);  mul = None
        mul_2 = torch.ops.aten.mul.Scalar(arg0_1, 0.125);  arg0_1 = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(add_1, [1], True)
        div_2 = torch.ops.aten.div.Scalar(sum_3, 2048.0);  sum_3 = None
        mul_3 = torch.ops.aten.mul.Scalar(div_2, 0.05);  div_2 = None
        expand_2 = torch.ops.aten.expand.default(mul_3, [256, 2048]);  mul_3 = None
        add_2 = torch.ops.aten.add.Tensor(add_1, expand_2);  add_1 = expand_2 = None
        return (add_2,)
        
