#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/3 下午9:03
# @Author  : liu yuhan
# @FileName: model.py
# @Software: PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
import sys
import os
from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score


def vector_matrix_T(arr, brr, brr_l2):
    return arr.dot(brr) / (np.sqrt(np.sum(arr * arr)) * brr_l2)


class TransE(nn.Module):

    def __init__(self, node_size, link_size, device, norm, dim, margin):
        super(TransE, self).__init__()
        self.node_size = node_size
        self.link_size = link_size
        self.device = device
        self.norm = norm
        self.dim = dim
        self.node_emb = self._init_node_emb()
        self.link_emb = self._init_link_emb()
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
        return self.loss(positive_distances, negative_distances)

    def loss(self, positive_distances, negative_distances):
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.criterion(positive_distances, negative_distances, target)

    def _distance(self, s, r, t):
        """
        Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id.
        """
        return (self.node_emb(s) + self.link_emb(r) - self.node_emb(t)).norm(p=self.norm, dim=1)


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
        return self.loss(positive_distances, negative_distances)

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
        h_trans = self._transfer(s_emb, r_norm)
        t_trans = self._transfer(t_emb, r_norm)

        return (h_trans + r_emb - t_trans).norm(p=self.norm, dim=1)


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
        return self.loss(positive_distances, negative_distances)

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
        # 超平面的法向量
        r_transfer = self.transfer_matrix(r)
        s_trans = self._transfer(s_emb, r_transfer)
        t_trans = self._transfer(t_emb, r_transfer)

        return (s_trans + r_emb - t_trans).norm(p=self.norm, dim=1)


# class TransD(nn.Module):
#
#     def __init__(self, node_size, link_size, device, norm, dim, margin):
#         super(TransD, self).__init__()
#         self.node_size = node_size
#         self.link_size = link_size
#         self.device = device
#         self.norm = norm
#         self.dim = dim
#         self.node_emb = self._init_node_emb()
#         self.link_emb = self._init_link_emb()
#         self.transfer_matrix = nn.Embedding(self.link_size, self.dim * self.dim)
#         self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')
#
#     def _init_node_emb(self):
#         node_emb = nn.Embedding(num_embeddings=self.node_size,
#                                 embedding_dim=self.dim)
#         uniform_range = 6 / np.sqrt(self.dim)
#         node_emb.weight.data.uniform_(-uniform_range, uniform_range)
#         return node_emb
#
#     def _init_link_emb(self):
#         link_emb = nn.Embedding(num_embeddings=self.link_size,
#                                 embedding_dim=self.dim)
#         uniform_range = 6 / np.sqrt(self.dim)
#         link_emb.weight.data.uniform_(-uniform_range, uniform_range)
#         return link_emb
#
#     def forward(self, sp, tp, sn, tn, r):
#         positive_distances = self._distance(sp, r, tp)
#         negative_distances = self._distance(sn, r, tn)
#         return self.loss(positive_distances, negative_distances)
#
#     def loss(self, positive_distances, negative_distances):
#         target = torch.tensor([-1], dtype=torch.long, device=self.device)
#         return self.criterion(positive_distances, negative_distances, target)
#
#     def _transfer(self, e, r_transfer):
#         r_transfer = r_transfer.view(-1, self.dim, self.dim)
#         e = e.view(-1, 1, self.dim)
#         e = torch.matmul(e, r_transfer)
#         return e.view(-1, self.dim)
#
#     def _distance(self, s, r, t):
#         """
#         Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id.
#         """
#         s_emb = self.node_emb(s)
#         t_emb = self.node_emb(t)
#         r_emb = self.link_emb(r)
#         r_transfer = self.transfer_matrix(r)
#         s_trans = self._transfer(s_emb, r_transfer)
#         t_trans = self._transfer(t_emb, r_transfer)
#
#         return (s_trans + r_emb - t_trans).norm(p=self.norm, dim=1)


class Evaluator:
    def __init__(self, model_save_path):
        self.score = 0
        self.status_best = []
        self.model_save_path = model_save_path

    def evaluate(self, epoch, model, test_list, loss):
        torch.set_grad_enabled(False)
        test_list = np.array(test_list)
        s_list, r_list, t_list = test_list[:, 0], test_list[:, 1], test_list[:, 2]
        score, hit10, hit3, hit1 = 0, 0, 0, 0
        node_emb = model.node_emb.weight.cpu().data.numpy()
        link_emb = model.link_emb.weight.cpu().data.numpy()
        node_emb_T = node_emb.T
        node_emb_L2 = np.sqrt(np.sum(node_emb * node_emb, axis=1))

        test_size = len(s_list)

        with tqdm(total=test_size) as bar:
            for s, r, t in zip(s_list, r_list, t_list):
                predict_t = node_emb[s] + link_emb[r]
                dis = vector_matrix_T(predict_t, node_emb_T, node_emb_L2)
                dis_top10 = np.argsort(-dis)[:10]
                index = list(np.where(dis_top10 == t)[0])
                if index:
                    index = index[0]
                    score += 1 / (index + 1)
                    if index == 0:
                        hit1 += 1
                        hit3 += 1
                        hit10 += 1
                    elif 0 < index < 3:
                        hit3 += 1
                        hit10 += 1
                    else:
                        hit10 += 1

                bar.set_description('Evaluate')
                bar.update(1)

        score, hit10, hit3, hit1 = score / test_size, hit10 / test_size, hit3 / test_size, hit1 / test_size

        status = ["epoch", epoch, "loss", loss, 'score', score,
                  'hit10:', hit10, 'hit3', hit3, 'hit1', hit1]
        print(status)
        if self.score < score:
            self.score = score
            self.status_best = status
            torch.save(model.state_dict(), self.model_save_path)

        torch.set_grad_enabled(True)
