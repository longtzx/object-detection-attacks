# -*-coding:utf-8-*-
from __future__ import division

import os
import xml.etree.ElementTree as ET
from utils.vis_tool import vis_bbox
import cv2
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
# from wider import WIDER
from skimage import transform as sktsf
from torch.autograd import Variable
import time

import attacks
from data.dataset import inverse_normalize
from data.dataset import pytorch_normalze
from data.util import read_image
from model import FasterRCNNVGG16
from trainer import BRFasterRcnnTrainer
from utils import array_tool as at

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
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

data_dir = '/home/yl/PycharmProjects/UEA/VOCdevkit/VOC2007'
attacker_path = 'checkpoints/10.path'
save_path_HOME = '/home/yl/PycharmProjects/UEA/result'


# save_path_HOME = '/home/xingxing/liangsiyuan/results/ssd_attack_video'


class VOCBboxDataset:
    """Bounding box dataset for PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`VOC_BBOX_LABEL_NAMES`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data.
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    """

    def __init__(self, data_dir, split='test',
                 use_difficult=False, return_difficult=False,
                 ):
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))
        # id_list_file = os.listdir(data_dir)
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES
        self.save_dir = save_path_HOME + '/frcnn/'
        self.save_dir_adv = save_path_HOME + '/JPEGImages/'
        self.save_dir_perturb = save_path_HOME + '/frcnn_perturb/'

    def __len__(self):
        return len(self.ids)

    def preprocess(self, img, min_size=600, max_size=600):
        """Preprocess an image for feature extraction.

        The length of the shorter edge is scaled to :obj:`self.min_size`.
        After the scaling, if the length of the longer edge is longer than
        :param min_size:
        :obj:`self.max_size`, the image is scaled to fit the longer edge
        to :obj:`self.max_size`.

        After resizing the image, the image is subtracted by a mean image value
        :obj:`self.mean`.

        Args:
            img (~numpy.ndarray): An image. This is in CHW and RGB format.
                The range of its value is :math:`[0, 255]`.
             (~numpy.ndarray): An image. This is in CHW and RGB format.
                The range of its value is :math:`[0, 255]`.

        Returns:
            ~numpy.ndarray:
            A preprocessed image.

        """
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

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        # print('id of img is:' + id_)
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            # if not self.use_difficult and int(obj.find('difficult').text) == 1:
            #     continue

            # difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(np.zeros(label.shape), dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)

        img = self.preprocess(img)
        img = torch.from_numpy(img)[None]
        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bbox, label, difficult, id_

    __getitem__ = get_example


def add_bbox(ax, bbox, label, score):
    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=2))

        caption = list()
        label_names = list(VOC_BBOX_LABEL_NAMES) + ['bg']
        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax


def img2jpg(img, img_suffix, quality):
    jpg_base = '/media/drive/ibug/300W_cropped/frcnn_adv_jpg/'
    img = img.transpose((1, 2, 0))
    img = Image.fromarray(img.astype('uint8'))
    if not os.path.exists(jpg_base):
        os.makedirs(jpg_base)
    jpg_path = jpg_base + img_suffix
    img.save(jpg_path, format='JPEG', subsampling=0, quality=quality)
    jpg_img = read_image(jpg_path)
    return jpg_img


if __name__ == '__main__':
    layer_idx = 20
    _data = VOCBboxDataset(data_dir)
    faster_rcnn = FasterRCNNVGG16()
    faster_rcnn.eval()
    attacker = attacks.Blade_runner()
    attacker.load(attacker_path)
    attacker.eval()
    trainer = BRFasterRcnnTrainer(faster_rcnn, attacker, \
                                  layer_idx=layer_idx, attack_mode=True).cuda()
    # trainer.load('/home/xlsy/Documents/CVPR19/final results/weights/fasterrcnn_img_0.701.pth')
    # trainer.load('/home/xingxing/liangsiyuan/code/weights/fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth')
    trainer.load(
        '/home/yl/PycharmProjects/UEA/checkpoints/fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth.701052458187')

    quality_list = [100, 90, 80, 70, 60, 50, 40, 30, 20]
    threshold = [0.7]
    adv_det_list = []

    for quality in threshold:

        img_detected_total = 0
        adv_detected_total = 0
        img_object_total = 0
        total_distance = 0
        total_time = 0
        trainer.faster_rcnn.score_thresh = quality

        print("数据的长度：", len(_data))

        for i in range(len(_data)):
        #     if i == 100:
        #         print('===================================')
        #         print(str(i) + ' imgs have been generated')
        #         print('===================================')
        #         break
            img_detected = 0
            adv_detected = 0
            img_object = 0

            # img,im_path = _data[i]
            img, _, img_labels, _, img_id = _data[i]
            print("img shape:", img.shape)

            img_labels_lists = np.unique(img_labels)
            img = Variable(img.float().cuda())

            rois, roi_scores = faster_rcnn(img, flag=True)

            rois_num = len(rois)
            if rois_num < 300:
                print(i)

            im_path_clone = str(img_id)
            # save_path = _data.save_dir + 'f0rcnn' + im_path.split('/')[-1]
            save_path = _data.save_dir + im_path_clone.split('/')[-1] + '.jpg'
            # print("图片", i, ":保存路径", save_path)
            save_path_adv = _data.save_dir_adv + im_path_clone.split('/')[-1] + '.jpg'
            save_path_perturb = _data.save_dir_perturb + 'frcnn_perturb_' + im_path_clone.split('/')[-1] + '.jpg'

            if not os.path.exists(_data.save_dir):
                os.makedirs(_data.save_dir)
            if not os.path.exists(_data.save_dir_adv):
                os.makedirs(_data.save_dir_adv)
            if not os.path.exists(_data.save_dir_perturb):
                os.makedirs(_data.save_dir_perturb)

            ori_img_ = inverse_normalize(at.tonumpy(img[0]))

            print("ori_img_ shape", ori_img_.shape)

            # _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_],\
            #         new_score=quality, visualize=True)
            before = time.time()
            adv_img, perturb, distance = trainer.attacker.perturb(img, save_perturb=save_path_perturb, rois=rois,
                                                                  roi_scores=roi_scores)
            print("save_path_perturb",save_path_perturb)
            print("adv_img shape:", adv_img.shape)

            after = time.time()
            generate_time = after - before
            total_time = total_time + generate_time
            total_distance = total_distance + distance
            adv_img_ = inverse_normalize(at.tonumpy(adv_img[0]))
            perturb_ = inverse_normalize(at.tonumpy(perturb[0]))
            del adv_img, perturb, img
            perturb_ = perturb_.transpose((1, 2, 0))

            # 将图片从BGR转换为RGB格式进行保存
            perturb_RGB = cv2.cvtColor(perturb_, cv2.COLOR_BGR2RGB)
            ori_img_RGB = cv2.cvtColor(ori_img_.transpose((1, 2, 0)), cv2.COLOR_BGR2RGB)
            adv_img_RGB = cv2.cvtColor(adv_img_.transpose((1, 2, 0)), cv2.COLOR_BGR2RGB)

            # cv2.imshow("Image", ori_img_RGB)
            # cv2.waitKey(0)

            cv2.imwrite(save_path_perturb, perturb_RGB) # 存放扰动的图片
            cv2.imwrite(save_path, ori_img_RGB)  # 原始图片存放在frcnn文件夹下
            cv2.imwrite(save_path_adv, adv_img_RGB) # 存放对抗过后的图片
            # print('The mean distance between ori and adv is %f' % (total_distance / i))
            #
            # #
            adv_bboxes, adv_labels, adv_scores = trainer.faster_rcnn.predict([adv_img_], \
                    new_score=quality, visualize=True, adv=True)
            #
            # print('generate adv for ', img_id)
            # _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], \
            #                                                         new_score=quality, visualize=True)
            # adv_bboxes, adv_labels, adv_scores = trainer.faster_rcnn.predict([adv_img_], \
            #                                                                  new_score=quality, visualize=True,
            #                                                                  adv=True)
            break
        print('generate adv for all imgs')
