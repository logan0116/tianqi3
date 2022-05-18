#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/16 下午8:18
# @Author  : liu yuhan
# @FileName: trans_r.py
# @Software: PyCharm


import torch
import torch.nn as nn
import numpy as np


class TransR(nn.Module):

    def __init__(self, node_size, link_size, device, norm, dim, margin):
        super(TransR, self).__init__()
        self.node_size = node_size
        self.link_size = link_size
        self.device = device
        self.norm = norm
        self.dim = dim
        self.node_emb = self._init_node_emb()
        self.link_emb = self._init_link_emb()
        self.transfer_matrix = self._init_transfer_matrix()
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
        transfer_matrix = nn.Embedding(num_embeddings=self.link_size,
                                       embedding_dim=self.dim * self.dim)
        uniform_range = 6 / self.dim
        transfer_matrix.weight.data.uniform_(-uniform_range, uniform_range)
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
        r_transfer = self.transfer_matrix(r)
        s_emb = self._transfer(s_emb, r_transfer)
        t_emb = self._transfer(t_emb, r_transfer)

        return (s_emb + r_emb - t_emb).norm(p=self.norm, dim=1)
