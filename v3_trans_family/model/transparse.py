#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/16 下午8:18
# @Author  : liu yuhan
# @FileName: transparse.py
# @Software: PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class transparse(nn.Module):
    def __init__(self, node_size, link_size, node_degree, device, norm, dim, margin, theta):
        super(transparse, self).__init__()
        self.node_size = node_size
        self.link_size = link_size
        self.device = device
        self.norm = norm
        self.dim = dim
        # 独有的参数
        self.node_degree = node_degree
        self.node_degree_max = np.max(node_degree, 0)
        self.node_degree = torch.LongTensor(self.node_degree)
        self.node_degree_max = torch.LongTensor(self.node_degree_max)

        self.theta = theta
        # 初始化
        self.node_emb = self._init_node_emb()
        self.link_emb = self._init_link_emb()
        self.s_transfer_matrix = self._init_transfer_matrix()
        self.t_transfer_matrix = self._init_transfer_matrix()
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

    def _transfer(self, node, link, e, m, is_source):
        m = m.view(-1, self.dim, self.dim)
        zero = torch.zeros_like(m).cpu().to(self.device)
        # 这里很重要的一个地方是稀疏化
        if is_source:
            # 头实体的出度
            theta = 1 - (1 - self.theta) * self.node_degree[node, link, 0] / self.node_degree_max[link, 0]
        else:
            # 尾实体的入度
            theta = 1 - (1 - self.theta) * self.node_degree[node, link, 1] / self.node_degree_max[link, 1]
        theta = theta * self.dim * self.dim
        theta = theta.view(-1, 1, 1).expand(-1, self.dim, self.dim).to(self.device)
        # 对m进行稀疏化，构建mask
        batch_size = theta.shape[0]
        # 生成一个长随机序列[B,dim*dim]
        mask = torch.cat([torch.randperm(self.dim * self.dim, dtype=torch.float32) for _ in range(batch_size)])
        # mask:[B, dim * dim] -> [B, dim, dim]
        mask = mask.view(batch_size, self.dim, self.dim).to(self.device)
        # mask中大于theta的部分置零
        mask = torch.where(mask > theta, zero, mask)
        # 对于m，mask中等于0的部分置零
        m = torch.where(mask == 0, zero, m)

        e = e.view(-1, self.dim, 1)
        e = torch.matmul(m, e)
        return e.view(-1, self.dim)

    def _distance(self, s, r, t):
        """
        Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id.
        """
        s_emb = self.node_emb(s)
        t_emb = self.node_emb(t)
        r_emb = self.link_emb(r)
        ms = self.s_transfer_matrix(r)
        mt = self.t_transfer_matrix(r)

        s_emb = self._transfer(s, r, s_emb, ms, is_source=True)
        t_emb = self._transfer(t, r, t_emb, mt, is_source=False)

        return (s_emb + r_emb - t_emb).norm(p=self.norm, dim=1)
