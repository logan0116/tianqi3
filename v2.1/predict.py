#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/4 上午4:18
# @Author  : liu yuhan
# @FileName: predict.py
# @Software: PyCharm


from utils import *
from model import *
from parser import *

import csv
import scipy.spatial.distance as distance


def get_link(link_path):
    link_list_origin = pd.read_csv(link_path, sep='\t', header=None).values.tolist()
    with open('../node2index.json', 'r', encoding='UTF-8') as file:
        node_dict = json.load(file)
    with open('../label2index.json', 'r', encoding='UTF-8') as file:
        label_dict = json.load(file)
    # 转换成索引
    link_list = [[node_dict[link[0]], label_dict[link[1]]] for link in link_list_origin]
    return link_list_origin, link_list


def index_reserve():
    with open('../node2index.json', 'r', encoding='UTF-8') as file:
        node_dict = json.load(file)
    return dict([(l2i[1], l2i[0]) for l2i in node_dict.items()])


if __name__ == '__main__':
    # 参数设置
    dim = 256
    batch_size = 128
    epochs = 10
    # 数据载入
    link_path = "../test.tsv"
    # 数据处理
    link_list_origin, link_list = get_link(link_path)
    # model load
    # cuda
    cuda_order = '0'
    device = torch.device("cuda:" + cuda_order if torch.cuda.is_available() else "cpu")
    print('device:', device)

    node_size, label_size = 27910, 136
    model_save_path = 'line-v1'
    trans_e = TransE(node_size, label_size, device, norm=1, dim=dim, margin=1.0)
    node_emb = trans_e.node_emb.weight.cpu().data.numpy()
    link_emb = trans_e.link_emb.weight.cpu().data.numpy()

    csv_writer = csv.writer(open('../test_label.tsv', 'w', encoding='UTF-8'), delimiter='\t')
    index2node = index_reserve()

    for link, link_origin in tqdm(zip(link_list, link_list_origin)):
        predict_t = node_emb[link[0]] + link_emb[link[1]]
        dis_dict = dict()
        for i in range(node_size):
            dis = distance.cosine(predict_t, node_emb[i])
            dis_dict[i] = dis
        dis_top10 = sorted(dis_dict.items(), key=lambda x: x[1], reverse=True)[1:11]
        csv_writer.writerow(link_origin + [index2node[node[0]] for node in dis_top10])
