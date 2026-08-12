
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class c6tjinw4ubtjay652a3f764zbgfuf7z7wbayxdoehx72axpmtwqt(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
        div = torch.ops.aten.div.Scalar(arg1_1, arg2_1);  arg1_1 = None
        expand = torch.ops.aten.expand.default(div, [arg3_1, arg2_1, arg4_1]);  div = None
        sub = torch.ops.aten.sub.Tensor(arg0_1, expand);  expand = None
        div_1 = torch.ops.aten.div.Scalar(arg5_1, arg2_1);  arg5_1 = None
        add = torch.ops.aten.add.Scalar(div_1, 0.0001);  div_1 = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        expand_1 = torch.ops.aten.expand.default(rsqrt, [arg3_1, arg2_1, arg4_1]);  rsqrt = arg3_1 = arg2_1 = arg4_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, expand_1);  sub = expand_1 = None
        mul_1 = torch.ops.aten.mul.Scalar(mul, 0.75);  mul = None
        mul_2 = torch.ops.aten.mul.Scalar(arg0_1, 0.25);  arg0_1 = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        return (add_1,)
        
