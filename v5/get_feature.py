#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/4 上午4:18
# @Author  : liu yuhan
# @FileName: predict.py
# @Software: PyCharm


import json
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from pytorch_pretrained_vit import ViT

from parser import *
from model_loader import *


def pil_loader(path):
    image = Image.open(path).convert('RGB')
    # 暴力resize
    resize = transforms.Resize([384, 384])
    image = resize(image)
    # loader
    loader = transforms.Compose([transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    return image


class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.model = ViT('B_16_imagenet1k', pretrained=True)

    def forward(self, x):
        x = self.model(x)
        return x


def get_image_feature():
    node_image_feature = []
    net = VisionTransformer().cuda()
    with open('../node2index.json', 'r', encoding='UTF-8') as file:
        node_dict = json.load(file)
    for node, node_index in tqdm(node_dict.items()):
        try:
            image = pil_loader('../images/' + node + '/image_0.jpg').cuda()
        except FileNotFoundError:
            image = torch.zeros((1, 3, 384, 384)).cuda()
        image_feature = net(image).reshape(-1, ).cpu().detach().numpy()
        node_image_feature.append(image_feature)
    np.save('node_image_feature', np.array(node_image_feature))


def get_trans_feature():
    # 参数设置
    args = parameter_parser()
    # cuda
    device = torch.device("cuda:0" + args.cuda_order if torch.cuda.is_available() else "cpu")
    # model load

    with open('../node2index.json', 'r', encoding='UTF-8') as file:
        node_dict = json.load(file)
    with open('../label2index.json', 'r', encoding='UTF-8') as file:
        label_dict = json.load(file)
    node_size = len(node_dict)
    label_size = len(label_dict)

    trans_model = model_load(args, node_size, label_size, device)
    trans_model.load_state_dict(torch.load(args.model + '/epoch-' + str(args.best_epoch_without_feature), map_location='cuda:0'))
    node_emb = trans_model.node_emb.weight.data.numpy()
    np.save('node_trans_feature', node_emb)


if __name__ == '__main__':
    get_image_feature()
    # get_trans_feature()
