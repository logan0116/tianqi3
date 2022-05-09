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


class GraphEncoder(nn.Module):

    def __init__(self, dim):
        super(GraphEncoder, self).__init__()
        # 这里引入一个简化的graph-embedding

        c_list_1 = [4, 32, 256]
        c_list_2 = [4, 16, 64, 256]

        self.cnn_1 = nn.Sequential(
            # ----------
            nn.Conv2d(in_channels=3, out_channels=c_list_1[0], kernel_size=10, stride=10, padding=0, bias=False),
            nn.BatchNorm2d(c_list_1[0]),
            nn.ReLU(inplace=True),
            # ----------
            nn.Conv2d(in_channels=c_list_1[0], out_channels=c_list_1[1], kernel_size=10, stride=10, padding=0,
                      bias=False),
            nn.BatchNorm2d(c_list_1[1]),
            nn.ReLU(inplace=True),
            # ----------
            nn.Conv2d(in_channels=c_list_1[1], out_channels=c_list_1[2], kernel_size=8, stride=8, padding=0,
                      bias=False),
            nn.BatchNorm2d(c_list_1[2]),
            nn.ReLU(inplace=True),
            # ----------
            nn.Conv2d(in_channels=c_list_1[2], out_channels=c_list_1[2], kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(c_list_1[2]),
        )

        self.cnn_2 = nn.Sequential(
            # ----------
            nn.Conv2d(in_channels=3, out_channels=c_list_2[0], kernel_size=8, stride=8, padding=0, bias=False),
            nn.BatchNorm2d(c_list_2[0]),
            nn.ReLU(inplace=True),
            # ----------
            nn.Conv2d(in_channels=c_list_2[0], out_channels=c_list_2[1], kernel_size=5, stride=5, padding=0,
                      bias=False),
            nn.BatchNorm2d(c_list_2[1]),
            nn.ReLU(inplace=True),
            # ----------
            nn.Conv2d(in_channels=c_list_2[1], out_channels=c_list_2[2], kernel_size=5, stride=5, padding=0,
                      bias=False),
            nn.BatchNorm2d(c_list_2[2]),
            nn.ReLU(inplace=True),
            # ----------
            nn.Conv2d(in_channels=c_list_2[2], out_channels=c_list_2[3], kernel_size=4, stride=4, padding=0,
                      bias=False),
            nn.BatchNorm2d(c_list_2[3]),
            nn.ReLU(inplace=True),
            # ----------
            nn.Conv2d(in_channels=c_list_2[3], out_channels=c_list_2[3], kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(c_list_2[3]),
        )
        self.relu = nn.ReLU(inplace=True)

        # liner
        self.liner = nn.Sequential(
            # ----------
            nn.Linear(in_features=dim, out_features=dim * 2),
            nn.ReLU(inplace=True),
            # ----------
            nn.Linear(in_features=dim * 2, out_features=dim),
            nn.ReLU(inplace=True))

    def encoder(self, x):
        # conv1层
        output_1 = self.cnn_1(x)  # torch.Size([1, 64, 56, 56])
        output_1 = self.relu(output_1)
        output_1 = torch.squeeze(output_1)
        output_2 = self.cnn_2(x)  # torch.Size([1, 64, 56, 56])
        output_2 = self.relu(output_2)
        output_2 = torch.squeeze(output_2)

        output = self.liner(output_1 + output_2)

        return output


class GraphTransE(nn.Module):

    def __init__(self, node_size, link_size, device, norm=1, dim=100, margin=1.0):
        super(GraphTransE, self).__init__()
        # 这里引入一个简化的resnet
        self.graph_encoder = GraphEncoder(dim)
        self.node_size = node_size
        self.link_size = link_size
        self.device = device
        self.norm = norm
        self.dim = dim
        self.link_emb = self._init_link_emb()
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')

    def _init_link_emb(self):
        link_emb = nn.Embedding(num_embeddings=self.link_size,
                                embedding_dim=self.dim)
        uniform_range = 6 / np.sqrt(self.dim)
        link_emb.weight.data.uniform_(-uniform_range, uniform_range)
        return link_emb

    def forward(self, sp, tp, sn, tn, r):
        """
        Return model losses based on the input.
        :param positive_triplets: triplets of positives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :param negative_triplets: triplets of negatives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: tuple of the model loss, positive triplets loss component, negative triples loss component
        """
        positive_distances = self._distance(sp, tp, r)
        negative_distances = self._distance(sn, tn, r)
        return self.loss(positive_distances, negative_distances)

    def predict(self, s, t, r):
        """
        Calculated dissimilarity score for given triplets.
        :param triplets: triplets in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: dissimilarity score for given triplets
        """
        return self._distance(s, t, r)

    def loss(self, positive_distances, negative_distances):
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.criterion(positive_distances, negative_distances, target)

    def _distance(self, s, t, r):
        """Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id."""
        return (self.graph_encoder.encoder(s) + self.link_emb(r) - self.graph_encoder.encoder(t)) \
            .norm(p=self.norm, dim=1)
