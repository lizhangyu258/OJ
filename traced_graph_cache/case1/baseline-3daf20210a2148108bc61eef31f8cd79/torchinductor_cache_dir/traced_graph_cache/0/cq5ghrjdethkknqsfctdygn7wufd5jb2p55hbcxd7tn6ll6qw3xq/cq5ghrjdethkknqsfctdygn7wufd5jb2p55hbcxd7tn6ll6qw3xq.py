
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class cq5ghrjdethkknqsfctdygn7wufd5jb2p55hbcxd7tn6ll6qw3xq(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
        mul = torch.ops.aten.mul.Tensor(arg0_1, arg1_1)
        sub = torch.ops.aten.sub.Tensor(arg0_1, arg1_1)
        add = torch.ops.aten.add.Tensor(mul, sub);  mul = sub = None
        expand = torch.ops.aten.expand.default(arg2_1, [256, 2048])
        add_1 = torch.ops.aten.add.Tensor(add, expand);  add = expand = None
        expand_1 = torch.ops.aten.expand.default(arg3_1, [256, 2048]);  arg3_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(add_1, expand_1);  add_1 = expand_1 = None
        expand_2 = torch.ops.aten.expand.default(arg2_1, [256, 2048]);  arg2_1 = None
        add_2 = torch.ops.aten.add.Tensor(arg0_1, expand_2);  arg0_1 = expand_2 = None
        relu = torch.ops.aten.relu.default(add_2);  add_2 = None
        mul_2 = torch.ops.aten.mul.Scalar(relu, 0.125);  relu = None
        add_3 = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        mul_3 = torch.ops.aten.mul.Scalar(arg1_1, 0.0625);  arg1_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_3, mul_3);  add_3 = mul_3 = None
        _to_copy = torch.ops.aten._to_copy.default(sub_1, dtype = torch.float32)
        sum_1 = torch.ops.aten.sum.dim_IntList(_to_copy, [1], True);  _to_copy = None
        div = torch.ops.aten.div.Scalar(sum_1, 2048.0);  sum_1 = None
        _to_copy_1 = torch.ops.aten._to_copy.default(div, dtype = torch.float16);  div = None
        expand_3 = torch.ops.aten.expand.default(_to_copy_1, [256, 2048]);  _to_copy_1 = None
        sub_2 = torch.ops.aten.sub.Tensor(sub_1, expand_3);  expand_3 = None
        mul_4 = torch.ops.aten.mul.Scalar(sub_2, 0.9);  sub_2 = None
        mul_5 = torch.ops.aten.mul.Scalar(sub_1, 0.1);  sub_1 = None
        add_4 = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        _to_copy_2 = torch.ops.aten._to_copy.default(add_4, dtype = torch.float32);  add_4 = None
        return (_to_copy_2,)
        
