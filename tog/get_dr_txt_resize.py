# ----------------------------------------------------#
#   获取测试集的detection-result和images-optional
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
# ----------------------------------------------------#
from models.frcnn import FRCNN
import os
from PIL import Image, ImageFont, ImageDraw
import torch as t
from misc_utils.util import read_image
from frcnn_utils import array_tool as at
from dataset_utils.preprocessing import letterbox_image_padded
from tog.attacks import *
from tqdm import tqdm
import time


class mAP_FRCNN(FRCNN):
    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image_id, image):
        # f = open("./input/detection-results/" + image_id + ".txt", "w")
        before = time.time()

        x_querys, x_meta = letterbox_image_padded(image, size=self.model_img_size)

        eps = 8 / 255.  # Hyperparameter: epsilon in L-inf norm
        eps_iter = 2 / 255.  # Hyperparameter: attack learning rate
        n_iter = 10  # Hyperparameter: number of attack iterations
        classes = self.classes
        model_img_size = self.model_img_size
        confidence_thresh_default = 0.05

        # x_adv_untargeted = tog_untargeted(victim=self, x_query=x_querys, n_iter=n_iter, eps=eps, eps_iter=eps_iter)
        x_adv_vanishing = tog_vanishing(victim=self, x_query=x_querys, n_iter=n_iter, eps=eps, eps_iter=eps_iter)

        after = time.time()
        generate_time = after - before
        # x_adv_fabrication = tog_fabrication(victim=self, x_query=x_querys, n_iter=n_iter, eps=eps, eps_iter=eps_iter)
        # x_adv_mislabeling_ml = tog_mislabeling(victim=self, x_query=x_querys, target='ml', n_iter=n_iter, eps=eps,eps_iter=eps_iter)
        # x_adv_mislabeling_ll = tog_mislabeling(victim=self, x_query=x_querys, target='ll', n_iter=n_iter, eps=eps, eps_iter=eps_iter)
        # detections_query = self.detect(x_adv_mislabeling_ml, conf_threshold=confidence_thresh_default)
        # x_query = x_adv_mislabeling_ml

        # print("x_querys: ", x_querys.shape)
        # print("x_adv_vanishing: ", x_adv_vanishing.shape)
        deta = x_querys[0] -  x_adv_vanishing[0]
        _l2 = np.linalg.norm(deta)
        l2 = int(99 * _l2 / np.linalg.norm(x_querys)) + 1
        # if len(x_query.shape) == 4:
        #     x_query = x_query[0]
        #
        # for box in detections_query:
        #     xmin = max(int(box[-4] * x_query.shape[1] / model_img_size[1]), 0)
        #     ymin = max(int(box[-3] * x_query.shape[0] / model_img_size[0]), 0)
        #     xmax = min(int(box[-2] * x_query.shape[1] / model_img_size[1]), x_query.shape[1])
        #     ymax = min(int(box[-1] * x_query.shape[0] / model_img_size[0]), x_query.shape[0])
        #
        #     predicted_class = classes[int(box[0])]
        #     sc = str(box[1])
        #     left = (xmin-x_meta[0])/x_meta[4]
        #     top = (ymin-x_meta[1])/x_meta[4]
        #     right = (xmax-x_meta[0])/x_meta[4]
        #     bottom = (ymax-x_meta[1])/x_meta[4]
        #     # 传出参数
        #     f.write("%s %s %s %s %s %s\n" % (predicted_class, sc[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
        # f.close()
        return generate_time, l2


weights = 'model_weights/FRCNN.pth'  # TODO: Change this path to the victim model's weights
frcnn = mAP_FRCNN().cuda(device=0).load(weights)
# image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()
image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()
if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")

total_distance = 0
total_time = 0
i = 0

for image_id in tqdm(image_ids):
    i = i + 1
    try:
        image_path = "./VOCdevkit/VOC2007/JPEGImages/" + image_id + ".jpg"
        image = Image.open(image_path)
        # image.save("./input/images-optional/" + image_id + ".jpg")
        getime, distance = frcnn.detect_image(image_id, image)

        total_distance = total_distance + distance

        total_time = total_time + getime
        # print("时间：", getime)
        # if (i == 1000):
        #     break

    # try:
    #     frcnn.detect_image(image_id, image)
    except Exception:
        pass
    # print(image_id, " done!")
print('The mean time  is %f' % (total_time / 4952))
print('The mean L2 distance between ori and adv is %f' % (total_distance / 4952))
print("Conversion completed!")
