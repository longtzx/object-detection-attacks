from dataset_utils.preprocessing import letterbox_image_padded
from misc_utils.visualization import visualize_detections
from models.frcnn import FRCNN
from PIL import Image
from tog.attacks import *
from misc_utils.vis_tool import vis_bbox
from misc_utils.util import read_image
import torch as t
from frcnn_utils import array_tool as at
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
from tqdm import tqdm
import time


weights = 'model_weights/FRCNN.pth'  # TODO: Change this path to the victim model's weights
detector = FRCNN().cuda(device=0).load(weights)

eps = 8 / 255.  # Hyperparameter: epsilon in L-inf norm
eps_iter = 2 / 255.  # Hyperparameter: attack learning rate
n_iter = 50  # Hyperparameter: number of attack iterations

fpath = './assets/000002.jpg'  # TODO: Change this path to the image to be attacked
img = read_image(fpath)
image = img.transpose((1, 2, 0))  # 还原
image = image[np.newaxis, :, :, :] / 255.
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

# 进行对抗
# x_adv_untargeted = tog_untargeted(victim=detector, x_query=image, n_iter=n_iter, eps=eps, eps_iter=eps_iter)
# x_adv_untargeted  = tog_vanishing(victim=detector, x_query=image, n_iter=n_iter, eps=eps, eps_iter=eps_iter)
# Visualizing the detection results on the adversarial example and compare them with that on the benign input
# detections_adv_untargeted = detector.detect(x_adv_untargeted, conf_threshold=detector.confidence_thresh_default)
# x_local = x_adv_untargeted.copy() * 255
# x_tensor = t.from_numpy(x_local[0].transpose((2, 0, 1)))[None]
image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()
total_time = 0
i = 0

for image_id in tqdm(image_ids):
    i = i + 1
    image_path = "./VOCdevkit/VOC2007/JPEGImages/" + image_id + ".jpg"

    before = time.time()
    img = read_image(image_path)
    image = img.transpose((1, 2, 0))  # 还原
    image = image[np.newaxis, :, :, :] / 255.
    img = t.from_numpy(img)[None]

    _bboxes, _labels, _scores, _logits = detector.faster_rcnn.predict(img , conf_threshold, iou_threshold)

    after = time.time()
    generate_time = after - before
    total_time = total_time + generate_time
    # vis_bbox(at.tonumpy(img[0]),
    #          at.tonumpy(_bboxes[0]),
    #          at.tonumpy(_labels[0]).reshape(-1),
    #          at.tonumpy(_scores[0]).reshape(-1))
    if (i == 1001):
        break
print('The mean time  is %f' % (total_time / 1000))