import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at

img = read_image('misc/demo.jpg')
img = t.from_numpy(img)[None]

faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()

# trainer.load(
#     '/home/yl/PycharmProjects/simple-faster-rcnn-pytorch-master/chainer_best_model_converted_to_pytorch_0.7053.pth')
trainer.load(
    '/home/yl/PycharmProjects/simple-faster-rcnn-pytorch-master/fasterrcnn_12222105_0.712649824453_caffe_pretrain.pth.712649824453')
opt.caffe_pretrain = True  # this model was trained from caffe-pretrained model
_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)
vis_bbox(at.tonumpy(img[0]),
         at.tonumpy(_bboxes[0]),
         at.tonumpy(_labels[0]).reshape(-1),
         at.tonumpy(_scores[0]).reshape(-1))
