#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/16 下午8:18
# @Author  : liu yuhan
# @FileName: trans_d.py
# @Software: PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransD(nn.Module):
    def __init__(self, node_size, link_size, device, norm, dim, margin):
        super(TransD, self).__init__()
        self.node_size = node_size
        self.link_size = link_size
        self.device = device
        self.norm = norm
        self.dim = dim
        self.node_emb = self._init_node_emb()
        self.link_emb = self._init_link_emb()
        self.node_transfer = self._init_node_emb()
        self.link_transfer = self._init_link_emb()
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

    def _transfer(self, e, e_transfer, r_transfer):
        """
        这里需要完成的是：h = (rp * hp + I) * h
        可以化简为：h = rp * (hp * h) + h
        因为我这里实体和连接的维数是一样的，所以这个部分还是相当简单的
        """
        e = e + torch.sum(e * e_transfer, -1, True) * r_transfer
        return F.normalize(e, p=2, dim=-1)

    def _distance(self, s, r, t):
        """
        Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id.
        """
        s_emb = self.node_emb(s)
        t_emb = self.node_emb(t)
        r_emb = self.link_emb(r)
        # 投影
        s_transfer = self.node_transfer(s)
        t_transfer = self.node_transfer(t)
        r_transfer = self.link_transfer(r)
        #
        s_emb = self._transfer(s_emb, s_transfer, r_transfer)
        t_emb = self._transfer(t_emb, t_transfer, r_transfer)
        return (s_emb + r_emb - t_emb).norm(p=self.norm, dim=1)
