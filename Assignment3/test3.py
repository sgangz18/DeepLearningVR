# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:15:12 2019

@author: Nikhil
"""

from __future__ import print_function
import torch

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

#print(z, out)
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
#print(a.requires_grad)
a.requires_grad_(True)
#print(a.requires_grad)
b = (a * a).sum()
#print(b.grad_fn)


x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)