from models.frcnn import FRCNN
from PIL import Image
from torch.autograd import Variable
import torch
import numpy as np
import os
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from PIL import Image, ImageFont, ImageDraw
import torch as t
from misc_utils.util import read_image
from frcnn_utils import array_tool as at
import copy
from dataset_utils.preprocessing import letterbox_image_padded

weights = 'model_weights/FRCNN.pth'  # TODO: Change this path to the victim model's weights
detector = FRCNN().cuda(device=0).load(weights)

fpath = './assets/000524.jpg'  # TODO: Change this path to the image to be attacked
img = read_image(fpath)
img = t.from_numpy(img)[None]

# 显示测试
# image = at.tonumpy(img[0])
# image = image.transpose((1, 2, 0))
# print(image.shape)
# plt.imshow(image.astype(np.uint8))
# plt.axis('off')
# plt.show()

# 传入网络
conf_threshold = 0.05
iou_threshold = 0.3
classes = detector.classes

_bboxes, _labels, _scores, _logits = detector.faster_rcnn.predict(img, conf_threshold, iou_threshold)
bbox = at.tonumpy(_bboxes[0])
label = at.tonumpy(_labels[0]).reshape(-1)
score = at.tonumpy(_scores[0]).reshape(-1)
for i, bb in enumerate(bbox):
    left = bb[1]
    top = bb[0]
    right = bb[3]
    bottom = bb[2]
    predicted_class = classes[int(label[i])]
    sc = str(score[i])
    print(predicted_class, sc[:6], str(left),str(int(top)), str(int(right)), str(int(bottom)))





# _bboxes = _bboxes[0][:, [1, 0, 3, 2]]  # ymin, xmin, ymax, xmax --> xmin, ymin, xmax, ymax
# _labels = np.expand_dims(_labels[0], axis=1)
# _scores = np.expand_dims(_scores[0], axis=1)
# _logits = _logits[0]
#
# detections_query = np.concatenate([_labels, _scores, _logits, _bboxes], axis=-1)
#
# for box in detections_query:
#     predicted_class = classes[int(box[0])]
#     score = str(box[1])
#     left, top, right, bottom = box[-4], box[-3], box[-2], box[-1]
#     # 传出参数
#     _bboxes = _bboxes[0][:, [1, 0, 3, 2]]
#     print("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

