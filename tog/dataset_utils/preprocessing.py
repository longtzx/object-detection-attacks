from PIL import Image
import numpy as np


def letterbox_image_padded(image, size=(416, 416)):
    """ Resize image with unchanged aspect ratio using padding """
    image_copy = image.copy()
    iw, ih = image_copy.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    # print("iw,ih", iw, ih)
    # print("w,h", w, h)
    # print("nw,nh", nw, nh)
    image_copy = image_copy.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0, 0, 0))
    new_image.paste(image_copy, ((w - nw) // 2, (h - nh) // 2))
    new_image = np.asarray(new_image)[np.newaxis, :, :, :] / 255.
    meta = ((w - nw) // 2, (h - nh) // 2, nw + (w - nw) // 2, nh + (h - nh) // 2, scale)

    return new_image, meta

def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height