#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/4 上午4:18
# @Author  : liu yuhan
# @FileName: predict.py
# @Software: PyCharm
import torch

from utils import *
from model_loader import *
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
    return link_list_origin, link_list


def index_reserve():
    with open('../node2index.json', 'r', encoding='UTF-8') as file:
        node_dict = json.load(file)
    return dict([(l2i[1], l2i[0]) for l2i in node_dict.items()])

def predict():
    # 参数设置
    args = parameter_parser()
    # eval
    # 数据处理
    print('data loading...')
    networkdeal = NetworkDeal(args.train_file_path)
    networkdeal.get_index()
    node_size, label_size, link_list = networkdeal.get_data()
    print('    node_size:', node_size, 'link_size:', len(link_list))
    _, test_list = data_split(link_list, rate=0.9)
    # 数据处理
    link_list_origin, link_list = get_link(args.test_file_path)
    # cuda
    device = torch.device("cuda:" + args.cuda_order if torch.cuda.is_available() else "cpu")
    print('device:', device)
    # model load
    print('model loading...')
    print('    model:', args.model)
    trans_model = model_load(args, node_size, label_size, device)
    trans_model.load_state_dict(torch.load(args.model + '/epoch-' + str(args.best_epoch), map_location='cuda:0'))
    print('    model load done.')
    # print('test...')
    # evaluator = Evaluator(args.model)
    # evaluator.evaluate(0, trans_model, test_list, 0)

    print('predict...')
    sr_list = trans_model.predict_sr(torch.LongTensor(link_list)[:, 0],
                                     torch.LongTensor(link_list)[:, 1]).detach().numpy()


    # 结果导出
    csv_writer = csv.writer(open('../test_label.tsv', 'w', encoding='UTF-8'), delimiter='\t')
    index2node = index_reserve()

    for i in tqdm(range(len(link_list))):
        sr = sr_list[i]
        t = trans_model.predict_t(torch.LongTensor([_ for _ in range(node_size)]),
                                  torch.LongTensor([link_list[i][1] for _ in range(node_size)])).detach().numpy()
        t_T = t.T
        t_L2 = np.sqrt(np.sum(t * t, axis=1))

        dis = vector_matrix_T(sr, t_T, t_L2)
        dis_top10 = np.argsort(-dis)[:10]
        csv_writer.writerow(link_list_origin[i] + [index2node[node] for node in dis_top10.tolist()])


if __name__ == '__main__':
    predict()