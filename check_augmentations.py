# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS, https://kaggle.com/zfturbo'

import os
import glob


if __name__ == '__main__':
    # Block to choose backend
    gpu_use = 0
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)
    print('GPU use: {}'.format(gpu_use))


import numpy as np
import cv2
from keras_augm_layer import *


def read_single_image(path):
    return cv2.imread(path)


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def check_augmentation_net(image_files, augm_per_image, output_directory):
    from keras.layers import Input
    from keras.models import Model

    images = []
    for f in image_files:
        img = read_single_image(f)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        images.append(img)
    images = np.array(images)
    print('Images shape: {}'.format(images.shape))

    transforms_all = [
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        RandomRotate(angle=45, p=0.5),
        RandomBrightness(p=0.999, max_delta=0.1),
        RGBShift(p=0.1, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
        ToGray(p=0.5),
        JpegCompression(p=0.9, quality_lower=5, quality_upper=99),
        RandomContrast(0.5, 1.5, p=0.9),
        RandomCrop(width_crop=(0.7, 1.0), height_crop=(0.7, 1.0), p=0.99),
        RandomHue(0.5, p=0.9),
        RandomSaturation(0.5, 1.5, p=0.9),
        RandomGaussNoise((10, 50), p=0.99),
        ResizeImage(width=256, height=256)
    ]

    transforms = transforms_all

    inp = Input((None, None, 3))
    x = AugmLayer(transforms, output_dim=(256, 256, 3), preproc_input=None)(inp, training=True)
    model = Model(inputs=inp, outputs=x)
    print(model.summary())

    for j in range(len(images)):
        if output_directory is None:
            show_image(images[j])
        batch = []
        for i in range(augm_per_image):
            batch.append(images[j])
        batch = np.array(batch, dtype=np.uint8)
        ret = model.predict(batch)
        for i in range(10):
            # print(ret.shape)
            if output_directory is None:
                show_image(ret[i])
            else:
                cv2.imwrite(os.path.join(output_directory, "img_{}_augm_{}.jpg".format(j, i)), ret[i])


if __name__ == '__main__':
    image_files = glob.glob("../input/*.jpg")
    augm_per_image = 10

    # If output_directory is None will show images on screen, otherwise write in specified directory
    output_directory = '../res/'
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    output_directory = None

    check_augmentation_net(image_files, augm_per_image, output_directory)
