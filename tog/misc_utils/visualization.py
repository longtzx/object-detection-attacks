import matplotlib.pyplot as plt
import numpy as np
import cv2


def visualize_detections(detections_dict):
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    # plt.clf()
    plt.figure(figsize=(3 * len(detections_dict), 3))
    for pid, title in enumerate(detections_dict.keys()):
        input_img, detections, model_img_size, classes, x_meta = detections_dict[title]
        # print(title)
        if len(input_img.shape) == 4:
            input_img = input_img[0]
        plt.subplot(1, len(detections_dict), pid + 1)
        plt.title(title)
        # print(input_img)
        plt.imshow(input_img)
        current_axis = plt.gca()
        for box in detections:
            xmin = max(int(box[-4] * input_img.shape[1] / model_img_size[1]), 0)
            ymin = max(int(box[-3] * input_img.shape[0] / model_img_size[0]), 0)
            xmax = min(int(box[-2] * input_img.shape[1] / model_img_size[1]), input_img.shape[1])
            ymax = min(int(box[-1] * input_img.shape[0] / model_img_size[0]), input_img.shape[0])
            color = colors[int(box[0])]
            print()
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='small', color='black', bbox={'facecolor': color, 'alpha': 1.0})
            predicted_class = classes[int(box[0])]
            sc = str(box[1])
            left = (xmin - x_meta[0]) / x_meta[4]
            top = (ymin - x_meta[1]) / x_meta[4]
            right = (xmax - x_meta[0]) / x_meta[4]
            bottom = (ymax - x_meta[1]) / x_meta[4]
            print(predicted_class, sc[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom)))
        plt.axis('off')
    plt.show()
