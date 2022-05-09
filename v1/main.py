#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/3 下午9:02
# @Author  : liu yuhan
# @FileName: main.py
# @Software: PyCharm

from utils import *
from model import *
from parser import *

if __name__ == '__main__':
    # 参数设置
    d = 256
    ratio = 1
    ng_num = 5
    batch_size = 128
    epochs = 10
    cuda_order = '0'
    # 数据载入
    link_path = "../train.tsv"
    # 数据处理
    print('data loading...')
    networkdeal = NetworkDeal(link_path, ng_num)
    networkdeal.get_network_feature()
    node_size, label_size, s_list, t_list, ng_list, label_list = networkdeal.get_data()
    loader = Data.DataLoader(MyDataSet(s_list, t_list, ng_list, label_list), batch_size, True)
    print('data load done.')

    # cuda
    device = torch.device("cuda:" + cuda_order if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # 模型初始化
    line = Line(ratio, node_size, label_size, d)
    line.to(device)
    optimizer = optim.Adam(line.parameters(), lr=0.00001)
    # 模型保存
    model_save_path = 'line-v1'

    # 保存平均的loss
    ave_loss = []
    loss_min = 100000
    loss_min_epoch = 0

    with tqdm(total=epochs) as bar:
        for epoch in range(epochs):
            loss_collector = []
            for i, (s, t, ng, y) in enumerate(loader):
                s, t, ng, y = s.to(device), t.to(device), ng.to(device), y.to(device)
                loss = line(s, t, ng, y)
                loss.backward()
                optimizer.step()
                loss_collector.append(loss.item())
            loss = np.mean(loss_collector)
            ave_loss.append(loss)
            bar.set_description('Epoch ' + str(epoch))
            bar.set_postfix(loss=loss)
            bar.update(1)
            if loss < loss_min:
                torch.save(line.state_dict(), model_save_path)
                loss_min = loss
                loss_min_epoch = epoch
    # loss_draw(epochs, ave_loss, loss_save_path, loss_min, loss_min_epoch)
