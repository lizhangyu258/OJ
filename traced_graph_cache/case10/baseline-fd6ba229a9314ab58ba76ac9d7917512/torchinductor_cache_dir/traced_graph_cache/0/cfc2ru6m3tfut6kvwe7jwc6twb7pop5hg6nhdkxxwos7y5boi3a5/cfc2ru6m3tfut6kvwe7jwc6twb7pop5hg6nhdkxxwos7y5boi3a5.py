
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class cfc2ru6m3tfut6kvwe7jwc6twb7pop5hg6nhdkxxwos7y5boi3a5(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1):
        _to_copy = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float32);  arg0_1 = None
        mul = torch.ops.aten.mul.Scalar(_to_copy, 0.3535533905932738);  _to_copy = None
        return (mul,)
        
