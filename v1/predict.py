#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/4 上午4:18
# @Author  : liu yuhan
# @FileName: predict.py
# @Software: PyCharm
import json

from utils import *
from model import *
from parser import *

import csv


def get_link(link_path):
    link_list_origin = pd.read_csv(link_path, sep='\t', header=None).values.tolist()
    with open('../node2index.json', 'r', encoding='UTF-8') as file:
        node_dict = json.load(file)
    # 转换成索引
    link_list = [[node_dict[link[0]], node_dict[link[1]]] for link in link_list_origin]
    return link_list_origin, link_list


def index_reserve():
    with open('../label2index.json', 'r', encoding='UTF-8') as file:
        label_dict = json.load(file)
    return dict([(l2i[1], l2i[0]) for l2i in label_dict.items()])


if __name__ == '__main__':
    # 参数设置
    d = 256
    ratio = 1
    ng_num = 5
    batch_size = 128
    epochs = 10
    cuda_order = '0'
    # 数据载入
    link_path = "../test.tsv"
    # 数据处理
    link_list_origin, link_list = get_link(link_path)
    # model load
    node_size, label_size = 27910, 136
    model_save_path = 'line-v1'
    line = Line(ratio, node_size, label_size, d)
    line.load_state_dict(torch.load(model_save_path))

    csv_writer = csv.writer(open('../test_label.tsv', 'w', encoding='UTF-8'), dialect='\t')
    index2label = index_reserve()

    for link, link_origin in zip(link_list, link_list_origin):
        label_index = line.predict(link[0], link[1])
        csv_writer.writerow([link_origin[0], index2label[int(label_index)], link_origin[1]])
