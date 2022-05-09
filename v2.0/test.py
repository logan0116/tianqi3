#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/4 下午3:43
# @Author  : liu yuhan
# @FileName: test.py
# @Software: PyCharm


import torch

local_heads = torch.LongTensor([1, 2, 3])

head_or_tail = torch.randint(high=2, size=local_heads.size())

print(head_or_tail)
random_entities = torch.randint(high=100, size=local_heads.size())
print(random_entities)
broken_heads = torch.where(head_or_tail == 1, random_entities, local_heads)
print(broken_heads)
# broken_tails = torch.where(head_or_tail == 0, random_entities, local_tails)
# negative_triples = torch.stack((broken_heads, local_relations, broken_tails), dim=1)


