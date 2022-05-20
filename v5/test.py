#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/4 下午3:43
# @Author  : liu yuhan
# @FileName: test.py
# @Software: PyCharm


# import torch
#
# local_heads = torch.LongTensor([1, 2, 3])
#
# head_or_tail = torch.randint(high=2, size=local_heads.size())
#
# print(head_or_tail)
# random_entities = torch.randint(high=100, size=local_heads.size())
# print(random_entities)
# broken_heads = torch.where(head_or_tail == 1, random_entities, local_heads)
# print(broken_heads)
# # broken_tails = torch.where(head_or_tail == 0, random_entities, local_tails)
# # negative_triples = torch.stack((broken_heads, local_relations, broken_tails), dim=1)

import numpy as np
import torch
import torch.nn.functional as F

#
# from torch.utils.data import random_split
# import torch.nn.functional as F
#
# a = torch.Tensor([2, 3, 4, 5])
# c = torch.Tensor([3, 4, 5, 6, 0, 1, 2, 3])
#
# b = a.view(2, 1, 2)
# c = c.view(2, 2, 2)
# print(c)
#
# error = torch.matmul(torch.matmul(b, c), b.permute((0, 2, 1)))
# print(error)
#
weight_2 = torch.Tensor([[1, 2, 3], [2, 3, 3], [9, 8, 3]])
weight_1 = torch.Tensor([[6, 7, 3], [2, 5, 3], [4, 8, 3]])
weight = torch.where(weight_2 > weight_1, weight_2, weight_1)
print(weight_1)
print(weight_2)
print(weight)
