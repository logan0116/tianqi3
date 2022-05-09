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


class TransE(nn.Module):

    def __init__(self, node_size, link_size, device, norm=1, dim=100, margin=1.0):
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
        # uniform_range = 6 / np.sqrt(self.dim)
        # entities_emb.weight.data.uniform_(-uniform_range, uniform_range)
        return node_emb

    def _init_link_emb(self):
        link_emb = nn.Embedding(num_embeddings=self.link_size,
                                embedding_dim=self.dim)
        # uniform_range = 6 / np.sqrt(self.dim)
        # relations_emb.weight.data.uniform_(-uniform_range, uniform_range)
        return link_emb

    def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor):
        """
        Return model losses based on the input.
        :param positive_triplets: triplets of positives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :param negative_triplets: triplets of negatives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: tuple of the model loss, positive triplets loss component, negative triples loss component
        """
        positive_distances = self._distance(positive_triplets)
        negative_distances = self._distance(negative_triplets)
        return self.loss(positive_distances, negative_distances)

    def predict(self, triplets: torch.LongTensor):
        """
        Calculated dissimilarity score for given triplets.
        :param triplets: triplets in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: dissimilarity score for given triplets
        """
        return self._distance(triplets)

    def loss(self, positive_distances, negative_distances):
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.criterion(positive_distances, negative_distances, target)

    def _distance(self, triplets):
        """Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id."""
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]
        return (self.node_emb(heads) + self.link_emb(relations) - self.node_emb(tails)).norm(p=self.norm, dim=1)
