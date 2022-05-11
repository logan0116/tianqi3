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
    args = parameter_parser()
    # 数据处理
    print('data loading...')
    networkdeal = NetworkDeal(args.train_file_path)
    networkdeal.get_index()
    node_size, label_size, link_list = networkdeal.get_data()
    train_list, test_list = data_split(link_list, rate=0.9)
    loader = Data.DataLoader(MyDataSet(train_list), args.batch_size, True)
    print('    data load done.')

    # cuda
    device = torch.device("cuda:" + str(args.cuda_order) if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # 模型初始化
    print('model loading...')
    print('    model:', args.model)
    trans_model = model_load(args, node_size, label_size, device)
    trans_model.to(device)
    optimizer = optim.Adam(trans_model.parameters(), lr=args.lr)
    print('    model load done.')

    # 模型评估
    evaluator = Evaluator(args.model)

    print('training...')
    for epoch in range(args.epochs):
        loss_collector = []
        with tqdm(total=len(loader)) as bar:
            for sp, r, tp in loader:
                # negatives make
                head_or_tail = torch.randint(high=2, size=r.size())
                random_entities = torch.randint(high=node_size, size=r.size())
                sn = torch.where(head_or_tail == 1, random_entities, sp)
                tn = torch.where(head_or_tail == 0, random_entities, tp)

                r = r.to(device)
                sp, tp = sp.to(device), tp.to(device)
                sn, tn = sn.to(device), tn.to(device)

                loss = trans_model(sp, tp, sn, tn, r).mean()
                loss.backward()
                optimizer.step()
                loss_collector.append(loss.item())
                bar.set_description('Epoch ' + str(epoch))
                bar.set_postfix(loss=loss)
                bar.update(1)
        if epoch % 5 == 0 and epoch > 0:
            evaluator.evaluate(epoch, trans_model, test_list, np.mean(loss_collector))

    print(evaluator.status_best)


if __name__ == '__main__':
    train()
