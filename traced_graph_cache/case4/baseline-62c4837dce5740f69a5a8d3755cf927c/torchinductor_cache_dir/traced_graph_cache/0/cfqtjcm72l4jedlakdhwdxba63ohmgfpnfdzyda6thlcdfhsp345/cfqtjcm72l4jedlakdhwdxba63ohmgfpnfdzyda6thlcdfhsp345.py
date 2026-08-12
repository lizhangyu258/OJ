
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class cfqtjcm72l4jedlakdhwdxba63ohmgfpnfdzyda6thlcdfhsp345(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
        expand = torch.ops.aten.expand.default(arg1_1, [256, 2048]);  arg1_1 = None
        add = torch.ops.aten.add.Tensor(arg0_1, expand);  arg0_1 = expand = None
        mul = torch.ops.aten.mul.Tensor(add, arg2_1);  add = arg2_1 = None
        expand_1 = torch.ops.aten.expand.default(arg3_1, [256, 2048]);  arg3_1 = None
        add_1 = torch.ops.aten.add.Tensor(mul, expand_1);  mul = expand_1 = None
        relu = torch.ops.aten.relu.default(add_1)
        mul_1 = torch.ops.aten.mul.Scalar(add_1, 0.25);  add_1 = None
        add_2 = torch.ops.aten.add.Tensor(relu, mul_1);  relu = mul_1 = None
        _to_copy = torch.ops.aten._to_copy.default(add_2, dtype = torch.float32)
        sum_1 = torch.ops.aten.sum.dim_IntList(_to_copy, [1], True);  _to_copy = None
        div = torch.ops.aten.div.Scalar(sum_1, 2048.0);  sum_1 = None
        _to_copy_1 = torch.ops.aten._to_copy.default(div, dtype = torch.float16);  div = None
        mul_2 = torch.ops.aten.mul.Scalar(_to_copy_1, 0.125);  _to_copy_1 = None
        expand_2 = torch.ops.aten.expand.default(mul_2, [256, 2048]);  mul_2 = None
        add_3 = torch.ops.aten.add.Tensor(add_2, expand_2);  add_2 = expand_2 = None
        _to_copy_2 = torch.ops.aten._to_copy.default(add_3, dtype = torch.float32);  add_3 = None
        return (_to_copy_2,)
        
