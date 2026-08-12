
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class c6on36u23tnopokg2ylumxybnetq4tnhy7bdayivbgyqrt3vkrso(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1):
        div = torch.ops.aten.div.Scalar(arg1_1, 1000.0);  arg1_1 = None
        add = torch.ops.aten.add.Tensor(arg0_1, div);  arg0_1 = div = None
        sum_1 = torch.ops.aten.sum.dim_IntList(add, [0]);  add = None
        div_1 = torch.ops.aten.div.Scalar(sum_1, 16.0);  sum_1 = None
        return (div_1,)
        
