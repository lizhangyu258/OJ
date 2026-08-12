
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class czfhyxv7osazsrhfyvyexya5f7kl7rmpa2apktrqp2siiv7bo5qp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1):
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(arg0_1, 2)
        sum_1 = torch.ops.aten.sum.dim_IntList(pow_1, [1], True);  pow_1 = None
        div = torch.ops.aten.div.Scalar(sum_1, 512.0);  sum_1 = None
        sqrt = torch.ops.aten.sqrt.default(div);  div = None
        add = torch.ops.aten.add.Scalar(sqrt, 1e-06);  sqrt = None
        expand = torch.ops.aten.expand.default(add, [32, 512]);  add = None
        div_1 = torch.ops.aten.div.Tensor(arg0_1, expand);  arg0_1 = expand = None
        expand_1 = torch.ops.aten.expand.default(arg1_1, [32, 512]);  arg1_1 = None
        mul = torch.ops.aten.mul.Tensor(div_1, expand_1);  div_1 = expand_1 = None
        return (mul,)
        
