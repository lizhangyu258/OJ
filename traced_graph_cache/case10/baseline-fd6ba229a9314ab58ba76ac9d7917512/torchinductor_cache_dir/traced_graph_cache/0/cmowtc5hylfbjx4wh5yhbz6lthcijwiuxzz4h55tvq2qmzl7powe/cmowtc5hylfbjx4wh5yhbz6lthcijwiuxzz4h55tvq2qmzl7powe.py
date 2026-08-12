
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class cmowtc5hylfbjx4wh5yhbz6lthcijwiuxzz4h55tvq2qmzl7powe(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1):
        view = torch.ops.aten.reshape.default(arg0_1, [4, 8, 128, 128]);  arg0_1 = None
        _to_copy = torch.ops.aten._to_copy.default(arg1_1, dtype = torch.float32);  arg1_1 = None
        expand = torch.ops.aten.expand.default(_to_copy, [4, 8, 128, 128]);  _to_copy = None
        add = torch.ops.aten.add.Tensor(view, expand);  view = expand = None
        amax = torch.ops.aten.amax.default(add, [3], True)
        expand_1 = torch.ops.aten.expand.default(amax, [4, 8, 128, 128]);  amax = None
        sub = torch.ops.aten.sub.Tensor(add, expand_1);  add = expand_1 = None
        exp = torch.ops.aten.exp.default(sub);  sub = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [3], True)
        expand_2 = torch.ops.aten.expand.default(arg2_1, [4, 8, 128, 128]);  arg2_1 = None
        full = torch.ops.aten.full.default([4, 8, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu'), pin_memory = False)
        expand_3 = torch.ops.aten.expand.default(sum_1, [4, 8, 128, 128]);  sum_1 = None
        div = torch.ops.aten.div.Tensor(exp, expand_3);  exp = expand_3 = None
        where = torch.ops.aten.where.self(expand_2, full, div);  expand_2 = full = div = None
        return (where,)
        
