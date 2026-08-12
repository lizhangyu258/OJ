
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class cvttrtwpecxu3nluhot4zns3ffojm6h4h342mgki55bhypoktfcz(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
        relu = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
        _to_copy = torch.ops.aten._to_copy.default(arg1_1, dtype = torch.float32);  arg1_1 = None
        expand = torch.ops.aten.expand.default(_to_copy, [arg2_1, arg3_1]);  _to_copy = None
        add = torch.ops.aten.add.Tensor(relu, expand);  relu = expand = None
        sum_1 = torch.ops.aten.sum.dim_IntList(add, [1], True)
        div = torch.ops.aten.div.Scalar(sum_1, arg3_1);  sum_1 = None
        mul = torch.ops.aten.mul.Scalar(div, 0.05);  div = None
        expand_1 = torch.ops.aten.expand.default(mul, [arg2_1, arg3_1]);  mul = arg2_1 = arg3_1 = None
        add_1 = torch.ops.aten.add.Tensor(add, expand_1);  add = expand_1 = None
        return (add_1,)
        
