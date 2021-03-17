# -*-coding:utf-8-*- # -*- coding: utf-8 -*-
# 改动：1.在 本文件88行；99行
#      2.attacks文件 213，加了dim

import numpy as np
import os
import ipdb
import torch
import torch.nn as nn
from utils.config import opt
from data.util import read_image
from data.dataset import Dataset
from data.dataset import inverse_normalize
from torch.utils import data as data_
from model import FasterRCNNVGG16
from functools import partial
from trainer import BRFasterRcnnTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from torch.autograd import Variable
from tqdm import tqdm
from data.dataset import preprocess, pytorch_normalze
import attacks
import warnings

warnings.filterwarnings("ignore")
layer_idies = [15, 20]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def store(model):
    """
    make hook for feature map
    """

    def hook(module, input, output, key):
        for layer_idx in layer_idies:
            if key == layer_idx:
                model.feature_maps[key] = output[0]

    for idx, layer in enumerate(model._modules.get('extractor')):
        layer.register_forward_hook(partial(hook, key=idx))


def calAttentionMask(rois, roi_scores, state=None, h=300, w=300):
    rois_num = len(roi_scores)
    ymins = []
    xmins = []
    ymaxs = []
    xmaxs = []

    if state == 0:
        # mask = np.zeros((300, 300), dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.float32)
    if state == 1:
        mask = np.zeros((75, 75), dtype=np.float32)
    if state == 2:
        mask = np.zeros((37, 37), dtype=np.float32)

    for i in range(rois_num):
        ymins.append(int(rois[i][0]))
        xmins.append(int(rois[i][1]))
        ymaxs.append(int(rois[i][2]))
        xmaxs.append(int(rois[i][3]))

    for i in range(rois_num):
        h = ymaxs[i] - ymins[i]
        w = xmaxs[i] - xmins[i]
        roi_weight = np.ones((h, w)) * roi_scores[i]
        if h == 0 or w == 0:
            mask = mask + 0
        else:
            mask[ymins[i]:ymaxs[i], xmins[i]:xmaxs[i]] = mask[ymins[i]:ymaxs[i], xmins[i]:xmaxs[i]] + roi_weight

    mask_min = np.min(mask)
    mask_max = np.max(mask)

    mask = (mask - mask_min) / (mask_max - mask_min)
    return mask


def train(**kwargs):
    opt._parse(kwargs)
    # opt.caffe_pretrain = True
    TrainResume = False
    dataset = Dataset(opt)
    # print('load dataset')
    dataloader = data_.DataLoader(dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=opt.num_workers)

    target_img_path = 'target_img.jpg'
    target_img = read_image(target_img_path) / 255
    target_img = torch.from_numpy(pytorch_normalze(target_img))
    target_img = torch.unsqueeze(target_img, 0).numpy()

    attacker = attacks.Blade_runner(train_BR=True)

    if TrainResume:
        # attacker.load('checkpoints/attack_02152100_0.path')
        attacker.load('checkpoints/attack_12131141_12.path')
    # attacker = attacks_no_target.Blade_runner(train_BR=True)
    faster_rcnn = FasterRCNNVGG16().eval()
    faster_rcnn.cuda()
    store(faster_rcnn)
    trainer = BRFasterRcnnTrainer(faster_rcnn, attacker, layer_idx=layer_idies, attack_mode=True).cuda()

    if opt.load_path:
        trainer.load(opt.load_path)
        # print('from %s Load model parameters' % opt.load_path)

    trainer.vis.text(dataset.db.label_names, win='labels')
    target_features_list = list()
    img_feature = trainer.faster_rcnn(torch.from_numpy(target_img).cuda())
    # print("特征图列表", len(img_feature))
    del img_feature
    target_features = trainer.faster_rcnn.feature_maps
    for target_feature_idx in target_features:
        target_features_list.append(target_features[target_feature_idx].cpu().detach().numpy())
    del target_features
    # print("特征图列表", len(target_features_list))
    for epoch in range(opt.epoch):
        trainer.reset_meters(BR=True)

        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            if isinstance(scale, torch.Tensor):
                scale = scale.cpu().numpy()
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            img, bbox, label = Variable(img), Variable(bbox), Variable(label)
            # print("\nscale:",scale)
            rois, roi_scores = faster_rcnn(img, flag=True)

            if len(rois) != len(roi_scores):
                print('The generated ROI and ROI score lengths are inconsistent')
            print(img.shape)
            print("scale", scale)
            print("bbox", bbox)

            gt_labels_np = label[0].cpu().detach().numpy()
            print("label", gt_labels_np)
            gt_labels_list = np.unique(gt_labels_np + 1)
            print("gt_labels_list", gt_labels_list)

            rois_feature1 = np.array(rois / 4, dtype=np.int32)
            rois_feature2 = np.array(rois / 8.1, dtype=np.int32)
            print("rois", rois.shape)
            print("roi_scores", roi_scores.shape)
            print("rois_feature1", rois_feature1.shape)
            print("rois_feature2", rois_feature2.shape)

            rois_feature1 = np.array(rois / 4, dtype=np.int32)
            rois_feature2 = np.array(rois / 8.1, dtype=np.int32)


            feature1_mask = calAttentionMask(rois_feature1, roi_scores, state=1)
            feature2_mask = calAttentionMask(rois_feature2, roi_scores, state=2)
            img_mask = calAttentionMask(rois, roi_scores, state=0)
            print("feature1_mask", feature1_mask.shape)
            print("feature2_mask", feature2_mask.shape)
            print("img_mask",img_mask.shape)

            img_mask = torch.from_numpy(np.tile(img_mask, (3, 1, 1))).cuda()
            feature1_mask = torch.from_numpy(np.tile(feature1_mask, (256, 1, 1))).cuda()
            feature2_mask = torch.from_numpy(np.tile(feature2_mask, (512, 1, 1))).cuda()
            print("修正后feature1_mask", feature1_mask.shape)
            print("修正后feature2_mask", feature2_mask.shape)
            print("修正后img_mask", img_mask.shape)
            break

        break


if __name__ == '__main__':
    train()
