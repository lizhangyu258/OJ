
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *

class c66xrh64berobvl5ebrigvxwyedx7wsdtl6ozndbdyb66lodgchl(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1):
        _to_copy = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float32);  arg0_1 = None
        _to_copy_1 = torch.ops.aten._to_copy.default(arg1_1, dtype = torch.float32);  arg1_1 = None
        expand = torch.ops.aten.expand.default(_to_copy_1, [256, 2048])
        add = torch.ops.aten.add.Tensor(_to_copy, expand);  expand = None
        sigmoid = torch.ops.aten.sigmoid.default(add);  add = None
        _to_copy_2 = torch.ops.aten._to_copy.default(arg2_1, dtype = torch.float32);  arg2_1 = None
        expand_1 = torch.ops.aten.expand.default(_to_copy_1, [256, 2048])
        sub = torch.ops.aten.sub.Tensor(_to_copy_2, expand_1);  expand_1 = None
        tanh = torch.ops.aten.tanh.default(sub);  sub = None
        mul = torch.ops.aten.mul.Tensor(sigmoid, tanh);  tanh = None
        add_1 = torch.ops.aten.add.Tensor(_to_copy, _to_copy_2);  _to_copy_2 = None
        expand_2 = torch.ops.aten.expand.default(_to_copy_1, [256, 2048]);  _to_copy_1 = None
        add_2 = torch.ops.aten.add.Tensor(add_1, expand_2);  add_1 = expand_2 = None
        relu = torch.ops.aten.relu.default(add_2);  add_2 = None
        mul_1 = torch.ops.aten.mul.Scalar(relu, 0.5)
        add_3 = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(sigmoid, relu);  sigmoid = relu = None
        mul_3 = torch.ops.aten.mul.Scalar(mul_2, 0.25);  mul_2 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_3, mul_3);  add_3 = mul_3 = None
        mul_4 = torch.ops.aten.mul.Scalar(_to_copy, 0.125);  _to_copy = None
        add_4 = torch.ops.aten.add.Tensor(sub_1, mul_4);  sub_1 = mul_4 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(add_4, [1], True)
        div = torch.ops.aten.div.Scalar(sum_1, 2048.0);  sum_1 = None
        mul_5 = torch.ops.aten.mul.Scalar(div, 0.2);  div = None
        expand_3 = torch.ops.aten.expand.default(mul_5, [256, 2048]);  mul_5 = None
        add_5 = torch.ops.aten.add.Tensor(add_4, expand_3);  add_4 = expand_3 = None
        return (add_5,)
        
