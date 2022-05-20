#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/16 下午8:18
# @Author  : liu yuhan
# @FileName: trans_a.py
# @Software: PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransA(nn.Module):

    def __init__(self, node_size, link_size, device, norm, dim, margin, L, C, lam):
        super(TransA, self).__init__()
        self.node_size = node_size
        self.link_size = link_size
        self.device = device
        self.norm = norm
        self.dim = dim
        self.L = L
        self.C = C
        self.lam = lam
        self.node_emb = self._init_node_emb()
        self.link_emb = self._init_link_emb()
        # Wr transA 核心的内容
        self.Wr = torch.zeros((self.link_size, self.dim, self.dim), device=self.device)

        self.margin = margin

    def _init_node_emb(self):
        node_emb = nn.Embedding(num_embeddings=self.node_size,
                                embedding_dim=self.dim)
        uniform_range = 6 / np.sqrt(self.dim)
        node_emb.weight.data.uniform_(-uniform_range, uniform_range)
        return node_emb

    def _init_link_emb(self):
        link_emb = nn.Embedding(num_embeddings=self.link_size,
                                embedding_dim=self.dim)
        uniform_range = 6 / np.sqrt(self.dim)
        link_emb.weight.data.uniform_(-uniform_range, uniform_range)
        return link_emb

    def calculateWr(self, sp_emb, tp_emb, sn_emb, tn_emb, r_emb, r):
        """
        对wr中的矩阵进行更新
        """
        error_p = torch.unsqueeze(torch.abs(sp_emb + r_emb - tp_emb), dim=1)
        error_n = torch.unsqueeze(torch.abs(sn_emb + r_emb - tn_emb), dim=1)
        # 讲道理，这个求和还是不能理解，测试一个不求和版本的。
        # self.Wr[r] += torch.sum(torch.matmul(error_n.permute((0, 2, 1)), error_n), dim=0) - \
        #               torch.sum(torch.matmul(error_p.permute((0, 2, 1)), error_p), dim=0)

        self.Wr[r] += torch.matmul(error_n.permute((0, 2, 1)), error_n) - \
                      torch.matmul(error_p.permute((0, 2, 1)), error_p)
        self.Wr[r] = F.relu(self.Wr[r])

    def _distance(self, s_emb, r_emb, t_emb, r):
        """
        Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id.
        """
        wr = self.Wr[r]
        # (B, E) -> (B, 1, E) * (B, E, E) * (B, E, 1) -> (B, 1, 1) -> (B, )
        error = torch.unsqueeze(torch.abs(s_emb + r_emb - t_emb), dim=1)
        error = torch.matmul(torch.matmul(error, wr), error.permute((0, 2, 1)))
        return torch.squeeze(error)

    def forward(self, sp, tp, sn, tn, r):
        size = sp.size()[0]
        # positive
        sp_emb = self.node_emb(sp)
        tp_emb = self.node_emb(tp)
        # negative
        sn_emb = self.node_emb(sn)
        tn_emb = self.node_emb(tn)
        r_emb = self.link_emb(r)

        # Wr更新
        self.calculateWr(sp_emb, tp_emb, sn_emb, tn_emb, r_emb, r)
        positive_distances = self._distance(sp_emb, r_emb, tp_emb, r)
        negative_distances = self._distance(sn_emb, r_emb, tn_emb, r)

        # Calculate loss
        margin_loss = 1 / size * torch.sum(F.relu(positive_distances - negative_distances + self.margin))
        wr_loss = 1 / self.link_size * torch.norm(input=self.Wr, p=self.L)
        weight_loss = 1 / self.node_size * torch.norm(input=self.node_emb.weight, p=self.L) + \
                      1 / self.link_size * torch.norm(input=self.link_emb.weight, p=self.L)

        print(margin_loss, wr_loss, weight_loss)
        print(margin_loss + self.lam * wr_loss + self.C * weight_loss)

        return margin_loss + self.lam * wr_loss + self.C * weight_loss
