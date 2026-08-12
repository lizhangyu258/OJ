
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class cjiz27aagpbp4qna2vqsgejbwetw2o6v4chcpwi27mterxslvpks(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1):
        sum_1 = torch.ops.aten.sum.dim_IntList(arg0_1, [1], True)
        div = torch.ops.aten.div.Scalar(sum_1, 512.0);  sum_1 = None
        expand = torch.ops.aten.expand.default(div, [32, 512])
        sub = torch.ops.aten.sub.Tensor(arg0_1, expand);  arg0_1 = expand = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(sub, 2)
        sum_2 = torch.ops.aten.sum.dim_IntList(pow_1, [1], True);  pow_1 = None
        div_1 = torch.ops.aten.div.Scalar(sum_2, 512.0);  sum_2 = None
        add = torch.ops.aten.add.Scalar(div_1, 1e-06);  div_1 = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        expand_1 = torch.ops.aten.expand.default(rsqrt, [32, 512])
        mul = torch.ops.aten.mul.Tensor(sub, expand_1);  sub = expand_1 = None
        expand_2 = torch.ops.aten.expand.default(arg1_1, [32, 512]);  arg1_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, expand_2);  mul = expand_2 = None
        expand_3 = torch.ops.aten.expand.default(arg2_1, [32, 512]);  arg2_1 = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, expand_3);  mul_1 = expand_3 = None
        return (div, rsqrt, add_1)
        
