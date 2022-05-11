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


def vector_matrix_T(arr, brr, brr_l2):
    return arr.dot(brr) / (np.sqrt(np.sum(arr * arr)) * brr_l2)


def get_link(link_path):
    link_list_origin = pd.read_csv(link_path, sep='\t', header=None).values.tolist()
    with open('../node2index.json', 'r', encoding='UTF-8') as file:
        node_dict = json.load(file)
    with open('../label2index.json', 'r', encoding='UTF-8') as file:
        label_dict = json.load(file)
    # 转换成索引
    link_list = [[node_dict[link[0]], label_dict[link[1]]] for link in link_list_origin]
    return len(node_dict), len(label_dict), link_list_origin, link_list


def index_reserve():
    with open('../node2index.json', 'r', encoding='UTF-8') as file:
        node_dict = json.load(file)
    return dict([(l2i[1], l2i[0]) for l2i in node_dict.items()])


if __name__ == '__main__':
    # 参数设置
    args = parameter_parser()
    # 数据处理
    node_size, label_size, link_list_origin, link_list = get_link(args.test_file_path)
    # cuda
    cuda_order = '0'
    device = torch.device("cuda:" + cuda_order if torch.cuda.is_available() else "cpu")
    print('device:', device)
    # model load
    print('model loading...')
    print('    model:', args.model)
    trans_model = model_load(args, node_size, label_size, device)
    trans_model.load_state_dict(torch.load(args.model + '/epoch-' + str(args.best_epoch)))
    print('    model load done.')
    node_emb = trans_model.node_emb.weight.data.numpy()
    link_emb = trans_model.link_emb.weight.data.numpy()
    node_emb_T = node_emb.T
    node_emb_L2 = np.sqrt(np.sum(node_emb * node_emb, axis=1))
    # 结果导出
    csv_writer = csv.writer(open('../test_label.tsv', 'w', encoding='UTF-8'), delimiter='\t')
    index2node = index_reserve()

    for link, link_origin in tqdm(zip(link_list, link_list_origin)):
        predict_t = node_emb[link[0]] + link_emb[link[1]]
        dis = vector_matrix_T(predict_t, node_emb_T, node_emb_L2)
        dis_top10 = np.argsort(-dis)[:10]
        csv_writer.writerow(link_origin + [index2node[node] for node in dis_top10.tolist()])
