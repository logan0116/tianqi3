#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/16 下午8:18
# @Author  : liu yuhan
# @FileName: trans_h.py
# @Software: PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransH(nn.Module):

    def __init__(self, node_size, link_size, device, norm, dim, margin):
        super(TransH, self).__init__()
        self.node_size = node_size
        self.link_size = link_size
        self.device = device
        self.norm = norm
        self.dim = dim
        self.node_emb = self._init_node_emb()
        self.link_emb = self._init_link_emb()
        self.norm_vector = self._init_link_emb()

        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')

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

    def forward(self, sp, tp, sn, tn, r):
        positive_distances = self._distance(sp, r, tp)
        negative_distances = self._distance(sn, r, tn)
        return torch.mean(self.loss(positive_distances, negative_distances))

    def loss(self, positive_distances, negative_distances):
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.criterion(positive_distances, negative_distances, target)

    def _transfer(self, e, norm):
        norm = F.normalize(norm, p=2, dim=-1)
        return e - torch.sum(e * norm, -1, True) * norm

    def _distance(self, s, r, t):
        """
        Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id.
        """
        s_emb = self.node_emb(s)
        t_emb = self.node_emb(t)
        r_emb = self.link_emb(r)
        # 超平面的法向量
        r_norm = self.norm_vector(r)
        s_emb = self._transfer(s_emb, r_norm)
        t_emb = self._transfer(t_emb, r_norm)

        return (s_emb + r_emb - t_emb).norm(p=self.norm, dim=1)
