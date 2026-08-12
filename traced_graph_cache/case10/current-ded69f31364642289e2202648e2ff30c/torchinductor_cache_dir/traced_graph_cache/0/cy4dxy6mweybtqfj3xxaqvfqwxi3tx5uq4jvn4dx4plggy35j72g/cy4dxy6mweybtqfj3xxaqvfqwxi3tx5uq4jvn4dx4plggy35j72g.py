
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class cy4dxy6mweybtqfj3xxaqvfqwxi3tx5uq4jvn4dx4plggy35j72g(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1):
        view = torch.ops.aten.reshape.default(arg0_1, [4, 8, 128, 64]);  arg0_1 = None
        _to_copy = torch.ops.aten._to_copy.default(view, dtype = torch.float16);  view = None
        mul = torch.ops.aten.mul.Scalar(arg1_1, 0.1);  arg1_1 = None
        add = torch.ops.aten.add.Tensor(_to_copy, mul);  mul = None
        mul_1 = torch.ops.aten.mul.Scalar(add, 0.75);  add = None
        relu = torch.ops.aten.relu.default(_to_copy);  _to_copy = None
        mul_2 = torch.ops.aten.mul.Scalar(relu, 0.25);  relu = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        _to_copy_1 = torch.ops.aten._to_copy.default(add_1, dtype = torch.float32)
        sum_1 = torch.ops.aten.sum.dim_IntList(_to_copy_1, [3], True);  _to_copy_1 = None
        div = torch.ops.aten.div.Scalar(sum_1, 64.0);  sum_1 = None
        _to_copy_2 = torch.ops.aten._to_copy.default(div, dtype = torch.float16);  div = None
        mul_3 = torch.ops.aten.mul.Scalar(_to_copy_2, 0.05);  _to_copy_2 = None
        expand = torch.ops.aten.expand.default(mul_3, [4, 8, 128, 64]);  mul_3 = None
        add_2 = torch.ops.aten.add.Tensor(add_1, expand);  add_1 = expand = None
        _to_copy_3 = torch.ops.aten._to_copy.default(add_2, dtype = torch.float32);  add_2 = None
        return (_to_copy_3,)
        
