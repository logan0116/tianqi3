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
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    image = Image.open(path).convert('RGB')
    # 暴力resize
    resize = transforms.Resize([800, 800])
    image = resize(image)
    # loader
    loader = transforms.Compose([transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    image = torch.squeeze(image)
    # print(image.shape)
    return image.detach().numpy()


#
#
# def node_trans(node_set):
#     node_set_trans = []
#     for node in node_set:
#         try:
#             node_set_trans.append(pil_loader('../images/' + node + '/image_0.jpg'))
#         except FileNotFoundError:
#             node_set_trans.append(np.zeros((3, 800, 800)))
#
#     return torch.Tensor(np.array(node_set_trans))


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
        # node_inf
        node_inf = []
        for node in tqdm(node_set):
            try:
                node_inf.append(pil_loader('../images/' + node + '/image_0.jpg'))
            except FileNotFoundError:
                node_inf.append(np.zeros((3, 800, 800)))

        return np.array(node_inf)

    def get_data(self):
        with open('../node2index.json', 'r', encoding='UTF-8') as file:
            node_dict = json.load(file)
        with open('../label2index.json', 'r', encoding='UTF-8') as file:
            label_dict = json.load(file)
        # 索引
        node_num = len(node_dict)
        label_num = len(label_dict)
        # 转换成索引
        s_list = [node_dict[link[0]] for link in self.link_list]
        r_list = [label_dict[link[1]] for link in self.link_list]
        t_list = [node_dict[link[2]] for link in self.link_list]
        return node_num, label_num, torch.LongTensor(s_list), torch.LongTensor(r_list), torch.LongTensor(t_list)


# class NetworkDeal:
#     """
#     这里需要生成很重要的一个结果是字典
#     """
#
#     def __init__(self, link_path):
#         self.link_list = pd.read_csv(link_path, sep='\t', header=None).values.tolist()
#         self.label_dict = dict()
#         self.node_set = []
#         self.node_num = 0
#         self.label_num = 0
#
#     def get_index(self):
#         node_set = set()
#         label_set = set()
#         for link in self.link_list:
#             node_set.add(link[0])
#             node_set.add(link[2])
#             label_set.add(link[1])
#         # entity
#         self.node_set = sorted(list(node_set))
#         # link
#         label_set = sorted(list(label_set))
#         self.label_dict = dict(zip(label_set, [i for i in range(len(label_set))]))
#
#         # 索引
#         self.node_num = len(node_set)
#         self.label_num = len(label_set)
#
#         with open('../label2index.json', 'w', encoding='UTF-8') as file:
#             json.dump(self.label_dict, file)
#
#     def get_data(self):
#         # 转换成索引
#         s_list = [link[0] for link in self.link_list]
#         r_list = [self.label_dict[link[1]] for link in self.link_list]
#         t_list = [link[2] for link in self.link_list]
#
#         return self.node_num, self.node_set, self.label_num, s_list, torch.LongTensor(r_list), t_list


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
