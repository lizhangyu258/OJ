
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class ctgqtwzn773hsmgjy3sjlgi5uzjuegxuxkrqhfnh3soizrqjm6ea(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1):
        _to_copy = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float32);  arg0_1 = None
        permute = torch.ops.aten.permute.default(_to_copy, [0, 1, 3, 2]);  _to_copy = None
        mul = torch.ops.aten.mul.Scalar(permute, 0.3535533905932738);  permute = None
        return (mul,)
        
