
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class c3ex66zro22754lthetx3cr62av2i2q3mhy55bhanlnjeznmmxpj(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
        permute = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        sigmoid = torch.ops.aten.sigmoid.default(arg1_1);  arg1_1 = None
        view = torch.ops.aten.reshape.default(sigmoid, [1, 1, 1, -1]);  sigmoid = None
        expand = torch.ops.aten.expand.default(view, [32, 64, 128, 16]);  view = None
        mul = torch.ops.aten.mul.Tensor(permute, expand);  permute = expand = None
        sum_1 = torch.ops.aten.sum.dim_IntList(mul, [3]);  mul = None
        permute_1 = torch.ops.aten.permute.default(arg2_1, [0, 2, 1]);  arg2_1 = None
        tanh = torch.ops.aten.tanh.default(arg3_1)
        mul_1 = torch.ops.aten.mul.Scalar(tanh, 0.125);  tanh = None
        add = torch.ops.aten.add.Scalar(mul_1, 1.0);  mul_1 = None
        view_1 = torch.ops.aten.reshape.default(add, [1, -1, 1]);  add = None
        expand_1 = torch.ops.aten.expand.default(view_1, [32, 64, 128]);  view_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(permute_1, expand_1);  permute_1 = expand_1 = None
        permute_2 = torch.ops.aten.permute.default(arg4_1, [0, 2, 1]);  arg4_1 = None
        sigmoid_1 = torch.ops.aten.sigmoid.default(arg3_1);  arg3_1 = None
        view_2 = torch.ops.aten.reshape.default(sigmoid_1, [1, -1, 1]);  sigmoid_1 = None
        expand_2 = torch.ops.aten.expand.default(view_2, [32, 64, 128])
        mul_3 = torch.ops.aten.mul.Tensor(permute_2, expand_2);  permute_2 = expand_2 = None
        add_1 = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        view_3 = torch.ops.aten.reshape.default(arg5_1, [1, 1, -1]);  arg5_1 = None
        expand_3 = torch.ops.aten.expand.default(view_3, [32, 64, 128]);  view_3 = None
        add_2 = torch.ops.aten.add.Tensor(add_1, expand_3);  add_1 = expand_3 = None
        exp = torch.ops.aten.exp.default(add_2);  add_2 = None
        add_3 = torch.ops.aten.add.Tensor(exp, sum_1);  exp = sum_1 = None
        relu = torch.ops.aten.relu.default(add_3);  add_3 = None
        mul_4 = torch.ops.aten.mul.Scalar(view_2, 0.015625);  view_2 = None
        add_4 = torch.ops.aten.add.Scalar(mul_4, 1.0);  mul_4 = None
        expand_4 = torch.ops.aten.expand.default(add_4, [32, 64, 128]);  add_4 = None
        mul_5 = torch.ops.aten.mul.Tensor(relu, expand_4);  relu = expand_4 = None
        permute_3 = torch.ops.aten.permute.default(mul_5, [0, 2, 1]);  mul_5 = None
        return (permute_3,)
        
