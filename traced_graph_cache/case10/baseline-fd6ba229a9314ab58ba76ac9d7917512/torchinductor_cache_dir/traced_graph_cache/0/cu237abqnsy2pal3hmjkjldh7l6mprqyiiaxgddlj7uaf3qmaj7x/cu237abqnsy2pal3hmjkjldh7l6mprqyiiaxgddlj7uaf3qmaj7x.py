
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class cu237abqnsy2pal3hmjkjldh7l6mprqyiiaxgddlj7uaf3qmaj7x(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1):
        view = torch.ops.aten.reshape.default(arg0_1, [4, 8, 128, 128]);  arg0_1 = None
        _to_copy = torch.ops.aten._to_copy.default(arg1_1, dtype = torch.float32);  arg1_1 = None
        expand = torch.ops.aten.expand.default(_to_copy, [4, 8, 128, 128]);  _to_copy = None
        add = torch.ops.aten.add.Tensor(view, expand);  view = expand = None
        eq = torch.ops.aten.eq.Scalar(add, -inf);  add = None
        return (eq,)
        
