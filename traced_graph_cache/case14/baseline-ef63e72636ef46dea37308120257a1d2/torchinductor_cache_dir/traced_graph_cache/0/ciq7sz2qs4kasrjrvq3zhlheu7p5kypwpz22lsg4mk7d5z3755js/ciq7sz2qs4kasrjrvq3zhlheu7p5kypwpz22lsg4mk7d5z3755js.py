
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class ciq7sz2qs4kasrjrvq3zhlheu7p5kypwpz22lsg4mk7d5z3755js(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
        div = torch.ops.aten.div.Scalar(arg1_1, arg2_1);  arg1_1 = None
        expand = torch.ops.aten.expand.default(div, [arg3_1, arg2_1, arg4_1]);  div = arg3_1 = arg2_1 = arg4_1 = None
        sub = torch.ops.aten.sub.Tensor(arg0_1, expand);  arg0_1 = expand = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(sub, 2);  sub = None
        return (pow_1,)
        
