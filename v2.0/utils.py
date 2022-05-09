#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/3 下午9:03
# @Author  : liu yuhan
# @FileName: utils.py
# @Software: PyCharm

import pandas as pd
import torch
import json
import torch.utils.data as Data


class NetworkDeal:
    """
    这里需要生成很重要的一个结果是字典
    """

    def __init__(self, link_path):
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
        self.node_num = len(node_dict)
        self.label_num = len(label_dict)
        with open('../node2index.json', 'w', encoding='UTF-8') as file:
            json.dump(node_dict, file)
        with open('../label2index.json', 'w', encoding='UTF-8') as file:
            json.dump(label_dict, file)

        # 转换成索引
        self.s_list = [node_dict[link[0]] for link in link_list]
        self.r_list = [label_dict[link[1]] for link in link_list]
        self.t_list = [node_dict[link[2]] for link in link_list]

    def get_data(self):
        return self.node_num, self.label_num, \
               torch.LongTensor(self.s_list), torch.LongTensor(self.r_list), torch.LongTensor(self.t_list)


class MyDataSet(Data.Dataset):
    '''
    没啥说的，正常的数据载入
    '''

    def __init__(self, s_list, r_list, t_list):
        self.s_list, self.r_list, self.t_list = s_list, r_list, t_list

    def __len__(self):
        return len(self.s_list)

    def __getitem__(self, idx):
        return self.s_list[idx], self.r_list[idx], self.t_list[idx]
