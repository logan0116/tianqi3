#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/3 下午9:03
# @Author  : liu yuhan
# @FileName: model_loader.py
# @Software: PyCharm


import torch
import numpy as np
import importlib
import os
from tqdm import tqdm


def model_load(args, node_size, label_size, device):
    model = importlib.import_module('model.' + args.model)
    if args.model == "trans_e":
        trans_model = model.TransE(node_size, label_size, device, norm=args.norm, dim=args.dim, margin=args.margin)
    elif args.model == "trans_h":
        trans_model = model.TransH(node_size, label_size, device, norm=args.norm, dim=args.dim, margin=args.margin)
    elif args.model == "trans_r":
        trans_model = model.TransR(node_size, label_size, device, norm=args.norm, dim=args.dim, margin=args.margin)
    elif args.model == "trans_d":
        trans_model = model.TransD(node_size, label_size, device, norm=args.norm, dim=args.dim, margin=args.margin,
                                   add_image_feature=args.add_image_feature)
    elif args.model == "trans_a":
        trans_model = model.TransA(node_size, label_size, device, norm=args.norm, dim=args.dim, margin=args.margin,
                                   L=args.L, C=args.C, lam=args.lam)
    # elif args.model == "trans_ad":
    #     trans_model = model.TransAD(node_size, label_size, device, norm=args.norm, dim=args.dim, margin=args.margin,
    #                                 L=args.L, C=args.C, lam=args.lam)
    # elif args.model == "transparse":
    #     trans_model = model.transparse(node_size, label_size, node_degree, device, norm=args.norm, dim=args.dim,
    #                                    margin=args.margin, theta=args.theta)

    else:
        raise ValueError('model not exist.')
    return trans_model


def vector_matrix_T(arr, brr, brr_l2):
    return arr.dot(brr) / (np.sqrt(np.sum(arr * arr)) * brr_l2)


class Evaluator:
    def __init__(self, model_save_path):
        self.score = 0
        self.status_best = []
        self.model_save_path = model_save_path
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)

    def evaluate(self, epoch, model, test_list, loss):
        torch.set_grad_enabled(False)
        test_list = np.array(test_list)
        s_list, r_list, t_list = test_list[:, 0], test_list[:, 1], test_list[:, 2]
        score, hit10, hit3, hit1 = 0, 0, 0, 0
        node_emb = model.node_emb.weight.cpu().data.numpy()
        link_emb = model.link_emb.weight.cpu().data.numpy()
        node_emb_T = node_emb.T
        node_emb_L2 = np.sqrt(np.sum(node_emb * node_emb, axis=1))

        test_size = len(s_list)

        with tqdm(total=test_size) as bar:
            for s, r, t in zip(s_list, r_list, t_list):
                predict_t = node_emb[s] + link_emb[r]
                dis = vector_matrix_T(predict_t, node_emb_T, node_emb_L2)
                dis_top10 = np.argsort(-dis)[:10]
                index = list(np.where(dis_top10 == t)[0])
                if index:
                    index = index[0]
                    score += 1 / (index + 1)
                    if index == 0:
                        hit1 += 1
                        hit3 += 1
                        hit10 += 1
                    elif 0 < index < 3:
                        hit3 += 1
                        hit10 += 1
                    else:
                        hit10 += 1

                bar.set_description('Evaluate')
                bar.update(1)

        score, hit10, hit3, hit1 = score / test_size, hit10 / test_size, hit3 / test_size, hit1 / test_size

        status = ["epoch", epoch, "loss", loss, 'score', score,
                  'hit10:', hit10, 'hit3', hit3, 'hit1', hit1]
        print(status)
        if self.score < score:
            self.score = score
            self.status_best = status
        torch.save(model.state_dict(), self.model_save_path + '/epoch-' + str(epoch))

        torch.set_grad_enabled(True)
