#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/3 下午9:03
# @Author  : liu yuhan
# @FileName: parser.py
# @Software: PyCharm

import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description='for cnc ner')
    # base
    parser.add_argument("--model", type=str, default='trans_r')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--cuda_order", type=str, default=0)
    # file
    parser.add_argument("--train_file_path", type=str, default="../train.tsv")
    parser.add_argument("--test_file_path", type=str, default="../test.tsv")
    # trans parameter
    parser.add_argument("--norm", type=int, default=1)
    parser.add_argument("--margin", type=float, default=1.0)
    # predict parameter
    parser.add_argument("--best_epoch", type=str, default=10)

    return parser.parse_args()
