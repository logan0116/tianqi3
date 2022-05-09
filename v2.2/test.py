# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # @Time    : 2022/5/4 下午3:43
# # @Author  : liu yuhan
# # @FileName: test.py
# # @Software: PyCharm
#
# import torch
# import torch.nn as nn
#
#
# class GraphEncoder(nn.Module):
#
#     def __init__(self):
#         super(GraphEncoder, self).__init__()
#         # 这里引入一个简化的graph-embedding
#
#         self.cnn_1 = nn.Sequential(
#             # ----------
#             nn.Conv2d(in_channels=3, out_channels=8, kernel_size=10, stride=10, padding=0, bias=False),
#             nn.BatchNorm2d(8),
#             nn.ReLU(inplace=True),
#             # ----------
#             nn.Conv2d(in_channels=8, out_channels=64, kernel_size=10, stride=10, padding=0, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # ----------
#             nn.Conv2d(in_channels=64, out_channels=512, kernel_size=8, stride=8, padding=0, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             # ----------
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(512),
#         )
#
#         self.cnn_2 = nn.Sequential(
#             # ----------
#             nn.Conv2d(in_channels=3, out_channels=8, kernel_size=8, stride=8, padding=0, bias=False),
#             nn.BatchNorm2d(8),
#             nn.ReLU(inplace=True),
#             # ----------
#             nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=5, padding=0, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             # ----------
#             nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, stride=5, padding=0, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             # ----------
#             nn.Conv2d(in_channels=128, out_channels=512, kernel_size=4, stride=4, padding=0, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             # ----------
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(512),
#         )
#         self.relu = nn.ReLU(inplace=True)
#
#         # liner
#         self.liner = nn.Sequential(
#             # ----------
#             nn.Linear(in_features=512, out_features=1024),
#             nn.ReLU(inplace=True),
#             # ----------
#             nn.Linear(in_features=1024, out_features=512),
#             nn.ReLU(inplace=True))
#
#     def graph_encoder(self, x):
#         # conv1层
#         output_1 = self.cnn_1(x)  # torch.Size([1, 64, 56, 56])
#         output_1 = self.relu(output_1)
#         output_1 = torch.squeeze(output_1)
#         output_2 = self.cnn_2(x)  # torch.Size([1, 64, 56, 56])
#         output_2 = self.relu(output_2)
#         output_2 = torch.squeeze(output_2)
#
#         output = self.liner(output_1 + output_2)
#
#         return output
#
#
# if __name__ == '__main__':
#     model = GraphTransE()
#     input = torch.randn(2, 3, 800, 800)
#     out = model.graph_encoder(input)
#     print(out.shape)

import numpy as np

# batch_size = 2
#
# node_set = ['a', 'b', 'c', 'd']
#
# s = ['a', 'c', ]
# print(s)
# head_or_tail = np.random.randint(2, size=batch_size)
# print(head_or_tail)
# random_entities = np.random.choice(node_set, size=2, replace=False)
# print(random_entities)
# sn = np.where(head_or_tail == 1, random_entities, s)
# print(sn)

a = np.array([1, 12, 13, 14, 5, 6, 7, 8])
print(a)
print(np.argsort(-a)[:5])

argsort_a = np.argsort(a)

print("argsort_a,从小到大的index:", argsort_a)
e = argsort_a[::-1]
print("最大的5位的index:", e[:5])
