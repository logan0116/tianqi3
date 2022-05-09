#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/3 下午9:02
# @Author  : liu yuhan
# @FileName: main.py
# @Software: PyCharm

from utils import *
from model import *
from parser import *

from torchsummary import summary


def train():
    args = parameter_parser()
    # 数据处理
    print('data loading...')
    networkdeal = NetworkDeal(args.train_file_path)
    networkdeal.get_index()
    node_size, node_set, label_size, s_list, r_list, t_list = networkdeal.get_data()
    loader = Data.DataLoader(MyDataSet(s_list, r_list, t_list), args.batch_size, True)
    print('data load done.')

    # cuda
    cuda_order = '0'
    device = torch.device("cuda:" + cuda_order if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # # 模型初始化
    trans_e = GraphTransE(node_size, label_size, device, norm=args.norm, dim=args.dim, margin=args.margin)
    summary(trans_e)
    trans_e.to(device)
    optimizer = optim.Adam(trans_e.parameters(), lr=0.00001)
    # 模型保存

    # 保存平均的loss
    ave_loss = []
    loss_min = 100000
    loss_min_epoch = 0

    for epoch in range(args.epochs):
        loss_collector = []
        with tqdm(total=len(loader)) as bar:
            for sp, r, tp in loader:
                # negatives make
                head_or_tail = np.random.randint(2, size=r.size())
                random_entities = np.random.choice(node_set, size=r.size(), replace=False)
                sn = np.where(head_or_tail == 1, random_entities, sp)
                tn = np.where(head_or_tail == 0, random_entities, tp)

                r = r.to(device)
                sp, tp = node_trans(sp).to(device), node_trans(tp).to(device)
                sn, tn = node_trans(sn).to(device), node_trans(tn).to(device)

                loss = trans_e(sp, tp, sn, tn, r).mean()
                loss.backward()
                optimizer.step()
                loss_collector.append(loss.item())
                bar.set_description('Epoch %d' % epoch)
                bar.set_postfix(loss=loss.item())
                bar.update(1)
            ave_loss.append(np.mean(loss_collector))

            if loss < loss_min:
                torch.save(trans_e.state_dict(), args.model_save_path)
                loss_min = loss
                loss_min_epoch = epoch

    print('min loss: %0.4f' % loss_min + 'min loss epoch: %d' % loss_min_epoch)


if __name__ == '__main__':
    train()
