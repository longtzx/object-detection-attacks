from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import numpy as np
import os, cv2
from rap import rap_utils
from models.frcnn import FRCNN_RAP
from rap.attack_fr_wrapper import FRAttackWrapper
from tqdm import tqdm

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


class mAP_FRCNN(FRCNN_RAP):
    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def rap_attack(self, image_id, path):

        # f = open("./input/detection-results/" + image_id + ".txt", "w")
        before = time.time()
        img = cv2.imread(path)
        # Original detection results as gt
        blobs = rap_utils.preproc_im(img)

        scores, boxes, rpn_cls_prob, rpn_bbox_pred_rec, rpn_bbox_pred = self.predict_all(blobs)
        real_bboxes_offset = rpn_bbox_pred.data.cpu().numpy()
        dets_all_cls = self.postproc_dets(scores, boxes)
        bboxes, labels, _ = self.convert_dets(dets_all_cls)

        noise = np.zeros(list(self.net._image.size()))
        blobs_ori = blobs
        for iter_n in range(aw.num_iter):

            _, _, rpn_cls_prob, rpn_bbox_pred_rec, rpn_bbox_offset = self.predict_all(blobs)
            inds_break = aw.search_candidates(bboxes, labels, rpn_cls_prob.data.cpu().numpy(),
                                              rpn_bbox_pred_rec)  # 真实的标签和真实的位置，rpn的前景和背景的标记，以及所有的rpn提取的proposals

            if len(inds_break) == 0 or iter_n == aw.num_iter - 1:
                break
            # print("剩余的数量：", len(inds_break))
            np_perturbed = aw.gen_adv(self.net._image, rpn_cls_prob, rpn_bbox_offset, real_bboxes_offset,
                                      inds_break)
            im_np = self.net._image.data.cpu().numpy() - np_perturbed
            blobs['data'] = np.transpose(im_np, (0, 2, 3, 1))
            noise += np_perturbed

            distort = aw.check_distortion(noise[0])
            # print('Distortion: ' + str(distort))
            if distort > aw.max_distortion:  # Jump if distortion is larger than threshold
                break
        # Vis
        noises = np.transpose(noise, (0, 2, 3, 1))[0]
        _l2 = np.linalg.norm(noises)
        l2 = int(99 * _l2 / np.linalg.norm(blobs_ori['data'][0])) + 1

        # print("noises", noises.shape)
        # print("data:", blobs['data'].shape)

        after = time.time()
        generate_time = after - before

        noise = np.transpose(noise[0], (1, 2, 0))
        vis_im = noise * 20 + 127
        # cv2.imwrite('./assets/noise.jpg', vis_im)

        scores, boxes = self.predict(blobs)
        dets_all_cls = self.postproc_dets(scores, boxes)
        bboxes, labels, det_scores = self.convert_dets(dets_all_cls)

        # for i in range(len(labels)):
        #     predicted_class = CLASSES[labels[i]]
        #     sc = str(det_scores[i])
        #
        #     bbox = bboxes[i]
        #
        #     left = bbox[0]
        #     top = bbox[1]
        #     right = bbox[2]
        #     bottom = bbox[3]
        #
        # # 传出参数
        # f.write("%s %s %s %s %s %s\n" % (predicted_class, sc[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
        # # print(predicted_class, " ", sc[:6], " ", str(int(left)), " ", str(int(top)), " ", str(int(right)), " ",
        # #       str(int(bottom)))
        # f.close()
        return generate_time, l2


saved_model = 'model_weights/vgg16_faster_rcnn_iter_110000.pth'
frcnn = mAP_FRCNN(saved_model).cuda(device=0)

aw = FRAttackWrapper()
# image_ids = open('VOCdevkit/VOC2007/ImageSets/Main1/test.txt').read().strip().split()
image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()
if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")

# image_ids = ["002412", "004983", "002509", "002954", "005869", "006229", "006763", "001240", "002808", "002082",
#              "005776", "001461", "007590", "006944", "006940"]

total_distance = 0
total_time = 0
i = 0
for image_id in tqdm(image_ids):
    i = i + 1
    try:
        image_path = "./VOCdevkit/VOC2007/JPEGImages/" + image_id + ".jpg"
        getime, l2 = frcnn.rap_attack(image_id, image_path)

        total_distance = total_distance + l2
        total_time = total_time + getime
    # try:
    #     time = frcnn.rap_attack(image_id, image_path)
    #     total_time = total_time+time
    except Exception:
        pass
    # # print(image_id, " done!")
    if (i == 1000):
        break

print('The mean time  is %f' % (total_time/1000))
print('The mean L2 distance between ori and adv is %f' % (total_distance/1000))
print("Conversion completed!")
