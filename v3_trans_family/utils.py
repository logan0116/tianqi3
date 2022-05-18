#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/3 下午9:03
# @Author  : liu yuhan
# @FileName: utils.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import torch
import json
import torch.utils.data as Data


def data_split(link_list, rate=0.9):
    """
    拆分训练集和验证集
    """
    data_size = len(link_list)
    train_size = int(rate * data_size)
    test_size = data_size - train_size
    train_list, test_list = Data.random_split(
        dataset=link_list,
        lengths=[train_size, test_size],
    )
    return train_list, test_list


class NetworkDeal:
    """
    这里需要生成很重要的一个结果是字典
    """

    def __init__(self, link_path):
        self.link_list = pd.read_csv(link_path, sep='\t', header=None).values.tolist()

    def get_index(self):
        """
        一次性的
        :return:
        """
        node_set = set()
        label_set = set()

        for link in self.link_list:
            node_set.add(link[0])
            node_set.add(link[2])
            label_set.add(link[1])
        # entity
        node_set = sorted(list(node_set))
        node_dict = dict(zip(node_set, [i for i in range(len(node_set))]))
        with open('../node2index.json', 'w', encoding='UTF-8') as file:
            json.dump(node_dict, file)
        # link
        label_set = sorted(list(label_set))
        label_dict = dict(zip(label_set, [i for i in range(len(label_set))]))
        with open('../label2index.json', 'w', encoding='UTF-8') as file:
            json.dump(label_dict, file)

    def get_degree(self):
        """
        TranSparse
        """
        with open('../node2index.json', 'r', encoding='UTF-8') as file:
            node_dict = json.load(file)
        with open('../label2index.json', 'r', encoding='UTF-8') as file:
            label_dict = json.load(file)
        # 索引
        node_num = len(node_dict)
        label_num = len(label_dict)
        node_degree = np.zeros((node_num, label_num, 2))
        for link in self.link_list:
            s = node_dict[link[0]]
            t = node_dict[link[2]]
            r = label_dict[link[1]]
            # out_degree
            node_degree[s][r][0] += 1
            # in_degree
            node_degree[t][r][1] += 1
        return node_degree

    def get_data(self):
        with open('../node2index.json', 'r', encoding='UTF-8') as file:
            node_dict = json.load(file)
        with open('../label2index.json', 'r', encoding='UTF-8') as file:
            label_dict = json.load(file)
        # 索引
        node_num = len(node_dict)
        label_num = len(label_dict)
        # 转换成索引
        link_list = [[node_dict[link[0]], label_dict[link[1]], node_dict[link[2]]] for link in self.link_list]
        return node_num, label_num, link_list


class MyDataSet(Data.Dataset):
    '''
    没啥说的，正常的数据载入
    '''

    def __init__(self, link_list):
        self.link_list = link_list

    def __len__(self):
        return len(self.link_list)

    def __getitem__(self, idx):
        return self.link_list[idx]
