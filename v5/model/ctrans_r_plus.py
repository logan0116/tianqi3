#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/16 下午8:18
# @Author  : liu yuhan
# @FileName: trans_r.py
# @Software: PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CTransR(nn.Module):

    def __init__(self, node_size, link_size, c, link_class, device, norm, alpha, dim, margin):
        super(CTransR, self).__init__()
        self.node_size = node_size
        self.link_size = link_size
        self.device = device
        self.norm = norm
        self.dim = dim
        # cluster number
        self.c = c
        # link_class
        self.link_class = link_class
        self.node_emb = self._init_node_emb()
        self.link_emb = self._init_link_emb()
        self.alpha = alpha
        # 构建一个和类相关的矩阵
        self.transfer_matrix = self._init_transfer_matrix()
        self.index_matrix = torch.arange(0, self.c * self.link_size)
        self.index_matrix = self.index_matrix.view(self.c, self.link_size)

        # init
        nn.init.xavier_uniform_(self.transfer_matrix.weight.data)
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

    def _init_transfer_matrix(self):
        transfer_matrix = nn.Embedding(num_embeddings=self.link_size * self.c,
                                       embedding_dim=self.dim * self.dim)
        return transfer_matrix

    def forward(self, sp, tp, sn, tn, r):
        positive_distances = self._distance(sp, r, tp)
        negative_distances = self._distance(sn, r, tn)
        return torch.mean(self.loss(positive_distances, negative_distances))

    def loss(self, positive_distances, negative_distances):
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.criterion(positive_distances, negative_distances, target)

    def _transfer(self, e, r_transfer):
        r_transfer = r_transfer.view(-1, self.dim, self.dim)
        e = e.view(-1, 1, self.dim)
        e = torch.matmul(e, r_transfer)
        return e.view(-1, self.dim)

    def _distance(self, s, r, t):
        """
        Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id.
        """
        s_emb = self.node_emb(s)
        t_emb = self.node_emb(t)
        r_emb = self.link_emb(r)
        r_class = self.link_class[r]
        # s_emb = F.normalize(s_emb, 2, -1)
        # t_emb = F.normalize(t_emb, 2, -1)
        # r_emb = F.normalize(r_emb, 2, -1)
        m_rc = self.transfer_matrix(self.index_matrix[r_class, r])

        s_emb = self._transfer(s_emb, m_rc)
        t_emb = self._transfer(t_emb, m_rc)
        r_emb

        return (s_emb + r_emb - t_emb).norm(p=self.norm, dim=1) + \
               self.alpha * (r_transfer - r_emb).norm(p=self.norm, dim=1)
