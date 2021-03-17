from model.test import _get_blobs
import numpy as np

def jaccard_np(boxes_a, boxes_b):
    """
    Compuate jaccard overlap
    :param boxes_a: Nx4
    :param boxes_b: Nx4
    :return:
    """
    boxes_a = np.array(boxes_a, dtype=np.float32)
    boxes_b = np.array(boxes_b, dtype=np.float32)
    max_xy = np.minimum(np.expand_dims(boxes_a[:, 2:], axis=1), np.expand_dims(boxes_b[:, 2:], axis=0))
    min_xy = np.maximum(np.expand_dims(boxes_a[:, :2], axis=1), np.expand_dims(boxes_b[:, :2], axis=0))

    inter = np.maximum((max_xy - min_xy), 0)
    inter_area = inter[:, :, 0] * inter[:, :, 1]

    area_a = np.expand_dims(((boxes_a[:, 2] - boxes_a[:, 0]) *
              (boxes_a[:, 3] - boxes_a[:, 1])), axis=1)
    area_b = np.expand_dims(((boxes_b[:, 2] - boxes_b[:, 0]) *
              (boxes_b[:, 3] - boxes_b[:, 1])), axis=0)
    union = area_a + area_b - inter_area
    return inter_area / union

def select_detections(detection_labels, detection_bboxes, target_bboxes, target_labels, threshold=0.3):
    """
    Select detections which have overlap larger than threshold
    :param detection_scores:
    :param detection_bboxes: xyxy
    :param threshold:
    :return:
    """
    inds_list = []
    # print("target_bbox_label", target_labels.shape)
    # print("target_bboxes:", target_bboxes.shape)
    for i, target_bbox in enumerate(target_bboxes):
        target_label = target_labels[i]
        inds = np.where(detection_labels == target_label)[0]
        IoU = jaccard_np([target_bbox], detection_bboxes[inds, :])[0]
        inds_IoU = inds[np.where(IoU > threshold)[0]]
        if len(inds_IoU):
          inds_list.append(inds_IoU)

    if inds_list == []:
        keep_inds = []
    else:
        keep_inds = np.unique(np.concatenate(inds_list))
    return keep_inds

def preproc_im(im):
    # Process image
    blobs, im_scales = _get_blobs(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    im_blob = blobs['data']
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
    return blobs
