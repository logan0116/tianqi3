#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/3 下午9:03
# @Author  : liu yuhan
# @FileName: utils.py
# @Software: PyCharm

import numpy as np
import networkx as nx
import pandas as pd
import torch
import json


class NetworkDeal:
    """
    这里需要生成很重要的一个结果是字典
    """

    def __init__(self, link_path, ng_num):
        self.ng_num = ng_num
        link_list = pd.read_csv(link_path, sep='\t', header=None).values.tolist()
        node_set = set()
        label_set = set()
        for link in link_list:
            node_set.add(link[0])
            node_set.add(link[2])
            label_set.add(link[1])
        node_set = sorted(list(node_set))
        label_set = sorted(list(label_set))
        # 索引
        node_dict = dict(zip(node_set, [i for i in range(len(node_set))]))
        label_dict = dict(zip(label_set, [i for i in range(len(label_set))]))
        with open('../node2index.json', 'w', encoding='UTF-8') as file:
            json.dump(node_dict, file)
        with open('../label2index.json', 'w', encoding='UTF-8') as file:
            json.dump(label_dict, file)

        # 转换成索引
        self.link_list = [[node_dict[link[0]], node_dict[link[2]]] for link in link_list]
        self.label_list = [label_dict[link[1]] for link in link_list]
        # 保存一个节点数
        self.node_num = len(node_dict)
        self.label_num = len(label_dict)
        print('num of node:', self.node_num)
        print('num of label:', self.label_num)
        print('num of link:', len(self.link_list))

    def get_network_feature(self):
        '''
        构建网络计算个重要参数——degree
        :return:
        '''
        g = nx.Graph()
        g.add_edges_from(self.link_list)
        self.node_degree_dict = dict(nx.degree(g))

    def get_data(self):
        '''
        负采样
        :return:
        '''
        # 负采样
        sample_table = []
        sample_table_size = 1e8
        # 词频0.75次方
        pow_frequency = np.array(list(self.node_degree_dict.values())) ** 0.75
        nodes_pow = sum(pow_frequency)
        ratio = pow_frequency / nodes_pow
        count = np.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            sample_table += [wid] * int(c)
        sample_table = np.array(sample_table)

        # 生成一个训练集：
        source_list = []
        target_list = []
        node_ng_list = []
        label_list = []

        for link, label in zip(self.link_list, self.label_list):
            source_list.append(link[0])
            target_list.append(link[1])
            node_ng = np.random.choice(sample_table, size=(self.ng_num)).tolist()
            node_ng_list.append(node_ng)
            label_list.append(label)

        return self.node_num, self.label_num, \
               torch.LongTensor(source_list), torch.LongTensor(target_list), \
               torch.LongTensor(node_ng_list), torch.LongTensor(label_list)
