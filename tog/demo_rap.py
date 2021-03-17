from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
from model.test import im_detect
# from model.nms_wrapper import nms
from torchvision.ops import nms
from for_test.utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, cv2

import argparse
from rap import rap_utils

from matplotlib import cm
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from models.frcnn import FRCNN_RAP
from rap.attack_fr_wrapper import FRAttackWrapper
import torch

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_%d.pth',), 'res101': ('res101_faster_rcnn_iter_%d.pth',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

COLORS = [cm.tab10(i) for i in np.linspace(0., 1., 10)]


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    # plt.draw()
    plt.show()


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(image_name)
    # print("im的形状：", im.shape)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    # im = im_names
    scores, boxes = im_detect(net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    thresh = 0.8

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    print(im.shape)
    cntr = -1

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(
            torch.from_numpy(cls_boxes), torch.from_numpy(cls_scores),
            NMS_THRESH)
        dets = dets[keep.numpy(), :]
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            continue
        else:
            cntr += 1

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1],
                              fill=False,
                              edgecolor=COLORS[cntr % len(COLORS)],
                              linewidth=3.5))
            ax.text(
                bbox[0],
                bbox[1] - 2,
                '{:s} {:.3f}'.format(cls, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14,
                color='white')

        ax.set_title(
            'All detections with threshold >= {:.1f}'.format(thresh),
            fontsize=14)

        plt.axis('off')
        plt.tight_layout()
    plt.show()


saved_model = 'model_weights/vgg16_faster_rcnn_iter_110000.pth'
# saved_model = 'model_weights/FRCNN.pth'

detector = FRCNN_RAP(saved_model).cuda(device=0)
aw = FRAttackWrapper()

demonet = 'vgg16'  # Network to use [vgg16 res101]
dataset = 'pascal_voc_0712'  # Trained dataset [pascal_voc pascal_voc_0712]


def rap_attack(path):
    img = cv2.imread(path)
    # Original detection results as gt
    blobs = rap_utils.preproc_im(img)

    scores, boxes, rpn_cls_prob, rpn_bbox_pred_rec, rpn_bbox_pred = detector.predict_all(blobs)
    real_bboxes_offset = rpn_bbox_pred.data.cpu().numpy()
    dets_all_cls = detector.postproc_dets(scores, boxes)
    bboxes, labels, det_scores = detector.convert_dets(dets_all_cls)

    for i in range(len(labels)):
        predicted_class = labels[i]
        sc = str(det_scores[i])

        bbox = bboxes[i]

        left = bbox[0]
        top = bbox[1]
        right = bbox[2]
        bottom = bbox[3]

        # 传出参数
        print(predicted_class, " ", sc[:6], " ", str(int(left)), " ", str(int(top)), " ", str(int(right)), " ",
              str(int(bottom)))
    print()
    vis_im = aw.vis(img, bboxes, labels, CLASSES)
    cv2.imwrite('./assets/005869_orig.jpg', vis_im)
    print("原始的结果", labels)

    noise = np.zeros(list(detector.net._image.size()))

    # print("迭代次数：", aw.num_iter)
    for iter_n in range(aw.num_iter):
        # print('iter: ' + str(iter_n))
        _, _, rpn_cls_prob, rpn_bbox_pred_rec, rpn_bbox_offset = detector.predict_all(blobs)

        inds_break = aw.search_candidates(bboxes, labels, rpn_cls_prob.data.cpu().numpy(), rpn_bbox_pred_rec)
        # print(inds_break)
        # print("剩余的数量：", len(inds_break))
        if len(inds_break) == 0 or iter_n == aw.num_iter - 1:
            break

        np_perturbed = aw.gen_adv(detector.net._image, rpn_cls_prob, rpn_bbox_offset, real_bboxes_offset, inds_break)
        im_np = detector.net._image.data.cpu().numpy() - np_perturbed
        blobs['data'] = np.transpose(im_np, (0, 2, 3, 1))
        noise += np_perturbed

        distort = aw.check_distortion(noise[0])
        # print('Distortion: ' + str(distort))
        if distort > aw.max_distortion:  # Jump if distortion is larger than threshold
            break
    # Vis
    noise = np.transpose(noise[0], (1, 2, 0))
    vis_im = noise * 20 + 127
    cv2.imwrite('./assets/noise.jpg', vis_im)

    scores, boxes = detector.predict(blobs)
    dets_all_cls = detector.postproc_dets(scores, boxes)
    bboxes, labels, _ = detector.convert_dets(dets_all_cls)
    print("对抗后的标签", labels)
    print("位置", np.floor(bboxes))
    print(blobs['im_info'][-1])


    img_resized = cv2.resize(img, None, None, fx=blobs['im_info'][-1], fy=blobs['im_info'][-1])
    vis_im = aw.vis(img_resized + noise, bboxes * blobs['im_info'][-1], labels, CLASSES)
    cv2.imwrite('./assets/005869.jpg_perturbed.jpg', vis_im)

    noise1 = cv2.resize(noise, None, None, fx=1 / blobs['im_info'][-1], fy=1 / blobs['im_info'][-1])
    vis_im1 = aw.vis(img + noise1, bboxes, labels, CLASSES)
    cv2.imwrite('./assets/small_005869.jpg', vis_im1)
    cv2.imwrite('./assets/fortest_005869.jpg',img + noise1)

    adv_img = img_resized + noise
    return adv_img


if True:
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    # model path
    # saved_model = os.path.join('../output', demonet, DATASETS[dataset][0], 'default',
    #                           NETS[demonet][0] %(70000 if dataset == 'pascal_voc' else 110000))

    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(21,
                            tag='default', anchor_scales=[8, 16, 32])

    net.load_state_dict(torch.load(saved_model, map_location=lambda storage, loc: storage))

    net.eval()
    if not torch.cuda.is_available():
        net._device = 'cpu'
    net.to(net._device)

    print('Loaded network {:s}'.format(saved_model))

    im_names = "./input/images-optional/002808.jpg"
    adv = rap_attack(im_names)
    # im = np.clip(adv.copy(), 0, 255).astype(np.uint8)
    # cv2.imwrite('./assets/adv_image.jpg',im)

    # im_names = "./assets/004545_perturbed.jpg"
    # # im_names = "./assets/adv_image.jpg"
    # demo(net, im_names)
