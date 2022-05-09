#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/3 下午9:02
# @Author  : liu yuhan
# @FileName: main.py
# @Software: PyCharm

from utils import *
from model import *
from parser import *


def train():
    # 参数设置
    dim = 256
    batch_size = 128
    epochs = 10
    # 数据载入
    link_path = "../train.tsv"
    # 数据处理
    print('data loading...')
    networkdeal = NetworkDeal(link_path)
    node_size, label_size, s_list, r_list, t_list = networkdeal.get_data()
    print(s_list[0])
    loader = Data.DataLoader(MyDataSet(s_list, r_list, t_list), batch_size, True)
    print('data load done.')

    # # cuda
    # cuda_order = '0'
    # device = torch.device("cuda:" + cuda_order if torch.cuda.is_available() else "cpu")
    # print('device:', device)
    #
    # # 模型初始化
    # trans_e = TransE(node_size, label_size, device, norm=1, dim=dim, margin=1.0)
    # trans_e.to(device)
    # optimizer = optim.Adam(trans_e.parameters(), lr=0.00001)
    # # 模型保存
    # model_save_path = 'line-v1'
    #
    # # 保存平均的loss
    # ave_loss = []
    # loss_min = 100000
    # loss_min_epoch = 0
    #
    # with tqdm(total=epochs) as bar:
    #     for epoch in range(epochs):
    #         loss_collector = []
    #         for s, r, t in loader:
    #             s, r, t = s.to(device), r.to(device), t.to(device)
    #
    #             positive_triples = torch.stack((s, r, t), dim=1)
    #             # negatives make
    #             head_or_tail = torch.randint(high=2, size=s.size(), device=device)
    #             random_entities = torch.randint(high=node_size, size=s.size(), device=device)
    #             broken_s = torch.where(head_or_tail == 1, random_entities, s)
    #             broken_t = torch.where(head_or_tail == 0, random_entities, t)
    #             negative_triples = torch.stack((broken_s, r, broken_t), dim=1)
    #
    #             loss = trans_e(positive_triples, negative_triples).mean()
    #             loss.backward()
    #             optimizer.step()
    #             loss_collector.append(loss.item())
    #         loss = np.mean(loss_collector)
    #         ave_loss.append(loss)
    #         bar.set_description('Epoch ' + str(epoch))
    #         bar.set_postfix(loss=loss)
    #         bar.update(1)
    #         if loss < loss_min:
    #             torch.save(trans_e.state_dict(), model_save_path)
    #             loss_min = loss
    #             loss_min_epoch = epoch
    #
    # print('min loss: %0.4f' % loss_min + 'min loss epoch: %d' % loss_min_epoch)


if __name__ == '__main__':
    train()
