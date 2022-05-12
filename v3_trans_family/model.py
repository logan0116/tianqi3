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


def model_load(args, node_size, label_size, device):
    if args.model == "trans_e":
        trans_model = TransE(node_size, label_size, device, norm=args.norm, dim=args.dim, margin=args.margin)
    elif args.model == "trans_h":
        trans_model = TransH(node_size, label_size, device, norm=args.norm, dim=args.dim, margin=args.margin)
    elif args.model == "trans_r":
        trans_model = TransR(node_size, label_size, device, norm=args.norm, dim=args.dim, margin=args.margin)
    elif args.model == "trans_a":
        trans_model = TransA(node_size, label_size, device, norm=args.norm, dim=args.dim, margin=args.margin,
                             L=args.L, C=args.C, lam=args.lam)
    elif args.model == "trans_ad":
        trans_model = TransAD(node_size, label_size, device, norm=args.norm, dim=args.dim, margin=args.margin,
                              L=args.L, C=args.C, lam=args.lam)
    else:
        raise ValueError('model not exist.')
    return trans_model


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
        s_emb = self.node_emb(s)
        t_emb = self.node_emb(t)
        r_emb = self.link_emb(r)

        return (s_emb + r_emb - t_emb).norm(p=self.norm, dim=1)


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
        s_emb = self._transfer(s_emb, r_norm)
        t_emb = self._transfer(t_emb, r_norm)

        return (s_emb + r_emb - t_emb).norm(p=self.norm, dim=1)


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
        r_transfer = self.transfer_matrix(r)
        s_emb = self._transfer(s_emb, r_transfer)
        t_emb = self._transfer(t_emb, r_transfer)

        return (s_emb + r_emb - t_emb).norm(p=self.norm, dim=1)


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
        return self.loss(positive_distances, negative_distances)

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
        self.Wr_replace = torch.zeros((self.link_size, self.dim, self.dim), device=self.device)

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
        self.Wr[r] += torch.sum(torch.matmul(error_n.permute((0, 2, 1)), error_n), dim=0) - \
                      torch.sum(torch.matmul(error_p.permute((0, 2, 1)), error_p), dim=0)

        self.Wr[r] = torch.where(self.Wr[r] < 0, self.Wr_replace[r], self.Wr[r])

        # self.Wr[r] += torch.matmul(error_n.permute((0, 2, 1)), error_n) - \
        #               torch.matmul(error_p.permute((0, 2, 1)), error_p)

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

        return margin_loss + self.lam * wr_loss + self.C * weight_loss


class TransAD(nn.Module):
    def __init__(self, node_size, link_size, device, norm, dim, margin, L, C, lam):
        super(TransAD, self).__init__()
        self.node_size = node_size
        self.link_size = link_size
        self.device = device
        self.norm = norm
        self.dim = dim
        self.node_emb = self._init_node_emb()
        self.link_emb = self._init_link_emb()
        self.node_transfer = self._init_node_emb()
        self.link_transfer = self._init_link_emb()
        self.L = L
        self.C = C
        self.lam = lam
        # Wr transA 核心的内容
        self.Wr = torch.zeros((self.link_size, self.dim, self.dim), device=self.device)
        self.Wr_replace = torch.zeros((self.link_size, self.dim, self.dim), device=self.device)

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

    def _transfer(self, e, e_transfer, r_transfer):
        """
        这里需要完成的是：h = (rp * hp + I) * h
        可以化简为：h = rp * (hp * h) + h
        因为我这里实体和连接的维数是一样的，所以这个部分还是相当简单的
        """
        e = e + torch.sum(e * e_transfer, -1, True) * r_transfer
        return F.normalize(e, p=2, dim=-1)

    def calculateWr(self, sp_emb, tp_emb, sn_emb, tn_emb, r_emb, r):
        """
        对wr中的矩阵进行更新
        """
        error_p = torch.unsqueeze(torch.abs(sp_emb + r_emb - tp_emb), dim=1)
        error_n = torch.unsqueeze(torch.abs(sn_emb + r_emb - tn_emb), dim=1)
        # 讲道理，这个求和还是不能理解，测试一个不求和版本的。
        self.Wr[r] += torch.sum(torch.matmul(error_n.permute((0, 2, 1)), error_n), dim=0) -\
                       torch.sum(torch.matmul(error_p.permute((0, 2, 1)), error_p), dim=0)

        self.Wr[r] = torch.where(self.Wr[r] < 0, self.Wr_replace[r], self.Wr[r])


        # self.Wr[r] += torch.matmul(error_n.permute((0, 2, 1)), error_n) - \
        #               torch.matmul(error_p.permute((0, 2, 1)), error_p)

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

        # 投影
        sp_transfer = self.node_transfer(sp)
        tp_transfer = self.node_transfer(tp)
        sn_transfer = self.node_transfer(sn)
        tn_transfer = self.node_transfer(tn)
        r_transfer = self.link_transfer(r)
        #
        sp_emb = self._transfer(sp_emb, sp_transfer, r_transfer)
        tp_emb = self._transfer(tp_emb, tp_transfer, r_transfer)
        sn_emb = self._transfer(sn_emb, sn_transfer, r_transfer)
        tn_emb = self._transfer(tn_emb, tn_transfer, r_transfer)

        # Wr更新
        self.calculateWr(sp_emb, tp_emb, sn_emb, tn_emb, r_emb, r)
        positive_distances = self._distance(sp_emb, r_emb, tp_emb, r)
        negative_distances = self._distance(sn_emb, r_emb, tn_emb, r)

        # Calculate loss
        margin_loss = 1 / size * torch.sum(F.relu(positive_distances - negative_distances + self.margin))
        wr_loss = 1 / self.link_size * torch.norm(self.Wr, p=self.L)
        weight_loss = 1 / self.node_size * torch.norm(self.node_emb.weight, p=self.L) + \
                      1 / self.link_size * torch.norm(self.link_emb.weight, p=self.L)

        return margin_loss + self.lam * wr_loss + self.C * weight_loss


class Evaluator:
    def __init__(self, model_save_path):
        self.score = 0
        self.status_best = []
        self.model_save_path = model_save_path
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)

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
        torch.save(model.state_dict(), self.model_save_path + '/epoch-' + str(epoch))

        torch.set_grad_enabled(True)
