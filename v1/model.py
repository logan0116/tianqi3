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
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
import sys
import os
from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score


class MyDataSet(Data.Dataset):
    '''
    没啥说的，正常的数据载入
    '''

    def __init__(self, s_list, t_list, ng_list, label_list):
        self.s_list = s_list
        self.t_list = t_list
        self.ng_list = ng_list
        self.label_list = label_list

    def __len__(self):
        return len(self.s_list)

    def __getitem__(self, idx):
        return self.s_list[idx], self.t_list[idx], self.ng_list[idx], self.label_list[idx]


class Line(nn.Module):
    def __init__(self, ratio, word_size, label_size, dim):
        super(Line, self).__init__()
        initrange = 0.5 / dim
        # line
        print('weight init...')
        self.ratio = ratio
        self.u_emd = nn.Embedding(word_size, dim)
        self.u_emd.weight.data.uniform_(-initrange, initrange)
        self.context_emd = nn.Embedding(word_size, dim)
        self.context_emd.weight.data.uniform_(-0, 0)
        # classify
        self.liner_1 = nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.LazyBatchNorm1d(dim * 2), nn.ReLU(True))
        self.liner_2 = nn.Sequential(nn.Linear(dim * 2, dim), nn.LazyBatchNorm1d(dim), nn.ReLU(True))
        self.liner_3 = nn.Sequential(nn.Linear(dim, label_size), nn.LazyBatchNorm1d(dim), nn.ReLU(True))
        self.fun_loss = nn.CrossEntropyLoss()

    # 这边进行一个一阶+二阶的
    def forward_line(self, s, t, ng):
        """
        :param s:
        :param t:
        :param ng:
        :return:
        """
        vector_i = self.u_emd(s)
        # 一阶
        vector_o1 = self.u_emd(t)
        vector_ng1 = self.u_emd(ng)
        output_1_1 = torch.matmul(vector_i, vector_o1.transpose(-1, -2)).squeeze()
        output_1_1 = F.logsigmoid(output_1_1)
        # 负采样的部分
        output_1_2 = torch.matmul(vector_i.unsqueeze(1), vector_ng1.transpose(-1, -2)).squeeze()
        output_1_2 = F.logsigmoid(-1 * output_1_2).sum(1)
        output_1 = -1 * (output_1_1 + output_1_2)
        # 二阶
        vector_o2 = self.context_emd(t)
        vector_ng2 = self.context_emd(ng)
        output_2_1 = torch.matmul(vector_i, vector_o2.transpose(-1, -2)).squeeze()
        output_2_1 = F.logsigmoid(output_2_1)
        # 负采样的部分
        output_2_2 = torch.matmul(vector_i.unsqueeze(1), vector_ng2.transpose(-1, -2)).squeeze()
        output_2_2 = F.logsigmoid(-1 * output_2_2).sum(1)

        # 组合
        output_2 = -1 * (output_2_1 + output_2_2)

        loss = torch.mean(output_1) + torch.mean(output_2)
        return loss

    def forward_classify(self, s, t, y):
        """
        :param s:
        :param t:
        :param y:
        :return:
        """
        vector_i = self.u_emd(s)
        vector_o = self.u_emd(t)
        output = self.liner_1(torch.cat((vector_i, vector_o), 1))
        output = self.liner_2(output)
        output = self.liner_3(output)
        loss = self.fun_loss(output, y)

        return loss

    def forward(self, s, t, ng, y):
        loss_1 = self.forward_line(s, t, ng)
        loss_2 = self.forward_classify(s, t, y)
        return self.ratio * loss_1 + loss_2

    def predict(self, s, t):
        vector_i = self.u_emd(s)
        vector_o = self.u_emd(t)
        output = self.liner_1(torch.cat((vector_i, vector_o), 1))
        output = self.liner_2(output)
        output = self.liner_3(output)
        output = torch.max(output, 1)[1]
        return output

# class Evaluator:
#     def __init__(self, model_save_path):
#         self.f1_best = 0
#         self.status_best = []
#         self.model_save_path = model_save_path
#
#     def evaluate(self, epoch, model, s, t, y, loss):
#         torch.set_grad_enabled(False)
#         out = model(test_x.cuda())
#         prediction = torch.max(out.cpu(), 1)[1]
#         pred_y = prediction.data.numpy()
#         target_y = test_y.data.numpy()
#
#         precision = precision_score(target_y, pred_y, average='weighted')
#         recall = recall_score(target_y, pred_y, average='weighted')
#         f1 = f1_score(target_y, pred_y, average='weighted')
#         status = ["epoch", epoch, "loss", loss, 'precision:', precision, 'recall:', recall, 'f1:', f1]
#         print(status)
#         if self.f1_best < f1:
#             self.f1_best = f1
#             self.status_best = status
#             torch.save(model.state_dict(), self.model_save_path)
#
#         torch.set_grad_enabled(True)
