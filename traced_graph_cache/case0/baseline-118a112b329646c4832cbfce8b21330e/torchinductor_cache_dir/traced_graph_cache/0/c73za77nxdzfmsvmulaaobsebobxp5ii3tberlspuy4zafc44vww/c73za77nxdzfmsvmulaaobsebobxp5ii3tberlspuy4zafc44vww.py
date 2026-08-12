
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class c73za77nxdzfmsvmulaaobsebobxp5ii3tberlspuy4zafc44vww(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1):
        mul = torch.ops.aten.mul.Tensor(arg0_1, arg1_1)
        sub = torch.ops.aten.sub.Tensor(arg0_1, arg1_1)
        add = torch.ops.aten.add.Tensor(mul, sub);  mul = sub = None
        mul_1 = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  arg0_1 = None
        mul_2 = torch.ops.aten.mul.Scalar(arg1_1, 0.5)
        add_1 = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        add_2 = torch.ops.aten.add.Scalar(add_1, 1.0);  add_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(add, add_2)
        add_3 = torch.ops.aten.add.Tensor(add, add_2);  add = add_2 = None
        mul_4 = torch.ops.aten.mul.Scalar(add_3, 0.1);  add_3 = None
        sub_1 = torch.ops.aten.sub.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
        mul_5 = torch.ops.aten.mul.Scalar(arg1_1, 1.5);  arg1_1 = None
        add_4 = torch.ops.aten.add.Tensor(sub_1, mul_5);  sub_1 = mul_5 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(add_4, [1])
        sum_2 = torch.ops.aten.sum.dim_IntList(add_4, [1]);  add_4 = None
        return (sum_1, sum_2)
        
