
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class cryzrhaaly7esjw53gfj3e3ij4modte3gcnfsvhdi4f6thcok5cb(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
        _to_copy = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float32);  arg0_1 = None
        _to_copy_1 = torch.ops.aten._to_copy.default(arg1_1, dtype = torch.float32);  arg1_1 = None
        expand = torch.ops.aten.expand.default(_to_copy_1, [arg2_1, arg3_1]);  _to_copy_1 = None
        add = torch.ops.aten.add.Tensor(_to_copy, expand);  expand = None
        _to_copy_2 = torch.ops.aten._to_copy.default(arg4_1, dtype = torch.float32);  arg4_1 = None
        expand_1 = torch.ops.aten.expand.default(_to_copy_2, [arg2_1, arg3_1]);  _to_copy_2 = None
        mul = torch.ops.aten.mul.Tensor(add, expand_1);  add = expand_1 = None
        relu = torch.ops.aten.relu.default(mul);  mul = None
        mul_1 = torch.ops.aten.mul.Scalar(_to_copy, 0.125);  _to_copy = None
        add_1 = torch.ops.aten.add.Tensor(relu, mul_1);  relu = mul_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(add_1, [1], True)
        div = torch.ops.aten.div.Scalar(sum_1, arg3_1);  sum_1 = None
        expand_2 = torch.ops.aten.expand.default(div, [arg2_1, arg3_1]);  div = arg2_1 = arg3_1 = None
        sub = torch.ops.aten.sub.Tensor(add_1, expand_2);  expand_2 = None
        mul_2 = torch.ops.aten.mul.Scalar(sub, 0.75);  sub = None
        mul_3 = torch.ops.aten.mul.Scalar(add_1, 0.25);  add_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        return (add_2,)
        
