
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class cvdits4mlc3jbvikhvtbgga6q5kfczwmy2gbhqhulyx7ypti4syf(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1):
        sum_1 = torch.ops.aten.sum.dim_IntList(arg0_1, [2], True)
        div = torch.ops.aten.div.Scalar(sum_1, 256.0);  sum_1 = None
        expand = torch.ops.aten.expand.default(div, [32, 128, 256]);  div = None
        sub = torch.ops.aten.sub.Tensor(arg0_1, expand);  arg0_1 = expand = None
        sum_2 = torch.ops.aten.sum.dim_IntList(sub, [2], True);  sub = None
        mul = torch.ops.aten.mul.Scalar(sum_2, 0.125)
        tanh = torch.ops.aten.tanh.default(sum_2);  sum_2 = None
        mul_1 = torch.ops.aten.mul.Scalar(tanh, 0.03125);  tanh = None
        add = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        expand_1 = torch.ops.aten.expand.default(add, [32, 128, 256]);  add = None
        sigmoid = torch.ops.aten.sigmoid.default(arg1_1);  arg1_1 = None
        view = torch.ops.aten.reshape.default(sigmoid, [1, 1, -1]);  sigmoid = None
        mul_2 = torch.ops.aten.mul.Scalar(view, 0.75)
        mul_3 = torch.ops.aten.mul.Scalar(view, 0.5);  view = None
        tanh_1 = torch.ops.aten.tanh.default(mul_3);  mul_3 = None
        mul_4 = torch.ops.aten.mul.Scalar(tanh_1, 0.125);  tanh_1 = None
        add_1 = torch.ops.aten.add.Tensor(mul_2, mul_4);  mul_2 = mul_4 = None
        expand_2 = torch.ops.aten.expand.default(add_1, [32, 128, 256]);  add_1 = None
        add_2 = torch.ops.aten.add.Tensor(expand_1, expand_2);  expand_1 = expand_2 = None
        mul_5 = torch.ops.aten.mul.Scalar(add_2, 0.875)
        tanh_2 = torch.ops.aten.tanh.default(add_2);  add_2 = None
        mul_6 = torch.ops.aten.mul.Scalar(tanh_2, 0.125);  tanh_2 = None
        add_3 = torch.ops.aten.add.Tensor(mul_5, mul_6);  mul_5 = mul_6 = None
        return (add_3,)
        
