
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class cl3lncaqr3bmgxltal44x4acay55bhzuez4bdddlnzdffjsb3byq(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1):
        relu = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
        _to_copy = torch.ops.aten._to_copy.default(arg1_1, dtype = torch.float32);  arg1_1 = None
        expand = torch.ops.aten.expand.default(_to_copy, [128, 512]);  _to_copy = None
        mul = torch.ops.aten.mul.Tensor(relu, expand);  relu = expand = None
        _to_copy_1 = torch.ops.aten._to_copy.default(arg2_1, dtype = torch.float32);  arg2_1 = None
        add = torch.ops.aten.add.Tensor(mul, _to_copy_1);  mul = None
        sum_1 = torch.ops.aten.sum.dim_IntList(add, [1], True)
        div = torch.ops.aten.div.Scalar(sum_1, 512.0);  sum_1 = None
        mul_1 = torch.ops.aten.mul.Scalar(div, 0.05);  div = None
        expand_1 = torch.ops.aten.expand.default(mul_1, [128, 512]);  mul_1 = None
        add_1 = torch.ops.aten.add.Tensor(add, expand_1);  add = expand_1 = None
        mul_2 = torch.ops.aten.mul.Scalar(_to_copy_1, 0.02);  _to_copy_1 = None
        add_2 = torch.ops.aten.add.Tensor(add_1, mul_2);  add_1 = mul_2 = None
        return (add_2,)
        
