from __future__ import division
import os
import numpy as np
from PIL import Image
import torch as t
from skimage import transform as sktsf
import attacks
from model import FasterRCNNVGG16
from trainer import BRFasterRcnnTrainer
from torch.autograd import Variable
from data.dataset import pytorch_normalze
from data.dataset import inverse_normalize
import ipdb
import pytorch_ssim
from utils import array_tool as at
import cv2
import time
from tqdm import tqdm,trange
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
data_dir = '/home/yl/PycharmProjects/UEA/VOCdevkit/VOC2007'
attacker_path = 'checkpoints/10.path'
save_path_HOME = '/home/yl/PycharmProjects/UEA/result'


def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file.

    This function reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.

    Args:
        path (str): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.

    Returns:
        ~numpy.ndarray: An image.
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def preprocess(img, min_size=600, max_size=600):
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    try:
        # img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect')
        img = sktsf.resize(img, (C, 300, 300), mode='reflect')
    except:
        ipdb.set_trace()
    # both the longer and shorter should be less than
    # max_size and min_size
    normalize = pytorch_normalze
    return normalize(img)


layer_idx = 20
faster_rcnn = FasterRCNNVGG16()
faster_rcnn.eval()
attacker = attacks.Blade_runner()
attacker.load(attacker_path)
attacker.eval()
trainer = BRFasterRcnnTrainer(faster_rcnn, attacker, \
                              layer_idx=layer_idx, attack_mode=True).cuda()
trainer.load(
    '/home/yl/PycharmProjects/UEA/checkpoints/fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth.701052458187')

quality_list = [100, 90, 80, 70, 60, 50, 40, 30, 20]
threshold = [0.7]
adv_det_list = []

image_ids = open('assert/4952/test-4952.txt').read().strip().split()
if not os.path.exists("./assert/detection-results"):
    os.makedirs("./assert/detection-results")


total_distance = 0
total_time = 0
i=0
for image_id in tqdm(image_ids):
    i = i+1
    f = open("./assert/detection-results/" + image_id + ".txt", "w")
    image_path = "./assert/JPEGImages-4952/" + image_id + ".jpg"

    img = read_image(image_path, color=True)
    c, h, w = img.shape
    # img = preprocess(img)
    img = img / 255
    normalize = pytorch_normalze
    img = normalize(img)

    img = t.from_numpy(img)[None]

    img = Variable(img.float().cuda())
    rois, roi_scores = faster_rcnn(img, flag=True)

    save_path_perturb = "./assert/images-adv/" + image_id + ".jpg"

    before = time.time()

    adv_img, perturb, distance = trainer.attacker.perturb(img, save_perturb=save_path_perturb, rois=rois,
                                                          roi_scores=roi_scores, h=h, w=w)
    after = time.time()
    generate_time = after - before
    total_time = total_time + generate_time

    total_distance = total_distance + distance

    # ori_img_ = inverse_normalize(at.tonumpy(img[0]))
    adv_img_ = inverse_normalize(at.tonumpy(adv_img[0]))

    # ori_img_RGB = cv2.cvtColor(ori_img_.transpose((1, 2, 0)), cv2.COLOR_BGR2RGB)
    # adv_img_RGB = cv2.cvtColor(adv_img_.transpose((1, 2, 0)), cv2.COLOR_BGR2RGB)

    # cv2.imwrite("./assert/ori_004545.jpg", ori_img_RGB)  # 原始图片存放在frcnn文件夹下
    # cv2.imwrite(save_path_perturb, adv_img_RGB)  # 存放对抗过后的图片

    # ######### 以下是重新测出其坐标，然后计算mAP
    # bboxes, labels, det_scores = trainer.faster_rcnn.predict([adv_img_], \
    #                                                                  new_score=0.05, visualize=True, adv=True)
    # bboxes = at.tonumpy(bboxes[0])
    # labels = at.tonumpy(labels[0]).reshape(-1)
    # det_scores = at.tonumpy(det_scores[0]).reshape(-1)
    # for i, bb in enumerate(bboxes):
    #
    #     predicted_class = classes[int(labels[i])]
    #     sc = str(det_scores[i])
    #
    #     left = bb[1]
    #     top = bb[0]
    #     right = bb[3]
    #     bottom = bb[2]
    #
    #     # 传出参数
    #     f.write("%s %s %s %s %s %s\n" % (
    #     predicted_class, sc[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
    #     # print(predicted_class, " ", sc[:6], " ", str(int(left)), " ", str(int(top)), " ", str(int(right)), " ",
    #     #       str(int(bottom)))
    # f.close()
    # print(image_id, " done!")
    # print("Psnr:", distance)
    # if(i == 1000):
    break
    # #######################################################################
print('The mean L2 distance between ori and adv is %f' % (total_distance/4952))
print('The mean time  is %f' % (total_time/4952))
print("Conversion completed!")
