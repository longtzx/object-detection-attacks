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


class mAP_FRCNN(FRCNN):
    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image_id, image, iou_threshold=0.05, conf_threshold=0.70):
        f = open("./input/detection-results/" + image_id + ".txt", "w")
        img = t.from_numpy(image)[None]

        classes = self.classes

        # 传入网络
        conf_threshold = 0.05
        iou_threshold = 0.3
        _bboxes, _labels, _scores, _logits = self.faster_rcnn.predict(img, conf_threshold, iou_threshold)

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

            # 传出参数
            f.write("%s %s %s %s %s %s\n" % (predicted_class, sc[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
        f.close()
        return

weights = 'model_weights/FRCNN.pth'  # TODO: Change this path to the victim model's weights
frcnn = mAP_FRCNN().cuda(device=0).load(weights)
image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")

for image_id in image_ids:
    image_path = "./VOCdevkit/VOC2007/JPEGImages/" + image_id + ".jpg"
    image = Image.open(image_path)
    image.save("./input/images-optional/" + image_id + ".jpg")
    img = read_image(image_path)
    # frcnn.detect_image(image_id, img)
    print(image_id, " done!")

print("Conversion completed!")
