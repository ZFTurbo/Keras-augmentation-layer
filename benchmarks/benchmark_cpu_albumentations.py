# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'

# Train MobileNet with batch generator and albumentations on CPU

import os
import glob

if __name__ == '__main__':
    # Block to choose backend
    gpu_use = 4
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)
    print('GPU use: {}'.format(gpu_use))


from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import pandas as pd
from multiprocessing.pool import ThreadPool
from keras_augm_layer import *
import cv2
import numpy as np
from albumentations import *


def filp_aug(p=1.0):
    return Compose([
        HorizontalFlip(p=1.0),
        VerticalFlip(p=1.0),
        ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=45, p=1.0, border_mode=cv2.BORDER_CONSTANT),
        Resize(height=INPUT_SHAPE[0], width=INPUT_SHAPE[1], always_apply=True)
    ], p=p)


def jpeg_compression_aug(p=1.0):
    return Compose([
        ImageCompression(p=1.0, quality_lower=5, quality_upper=99),
        Resize(height=INPUT_SHAPE[0], width=INPUT_SHAPE[1], always_apply=True)
    ], p=p)


def color_noise_aug(p=1.0):
    return Compose([
        RandomBrightnessContrast(p=1.0, brightness_limit=0.2, contrast_limit=0.0),
        RandomBrightnessContrast(p=1.0, contrast_limit=0.2, brightness_limit=0.0),
        RGBShift(p=1.0, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0, val_shift_limit=0, p=1.0),
        HueSaturationValue(hue_shift_limit=0, sat_shift_limit=30, val_shift_limit=0, p=1.0),
        GaussNoise((10, 50), p=1.0),
        Resize(height=INPUT_SHAPE[0], width=INPUT_SHAPE[1], always_apply=True)
    ], p=p)


def many_aug(p=1.0):
    return Compose([
        HorizontalFlip(p=1.0),
        VerticalFlip(p=1.0),
        RandomRotate90(p=1.0),
        ShiftScaleRotate(p=1.0, shift_limit=0.0, scale_limit=0.0, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT),
        RandomBrightnessContrast(p=1.0, brightness_limit=0.2, contrast_limit=0.0),
        RandomBrightnessContrast(p=1.0, contrast_limit=0.2, brightness_limit=0.0),
        RGBShift(p=1.0, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
        ToGray(p=1.0),
        ImageCompression(p=1.0, quality_lower=5, quality_upper=99),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0, val_shift_limit=0, p=1.0),
        HueSaturationValue(hue_shift_limit=0, sat_shift_limit=30, val_shift_limit=0, p=1.0),
        GaussNoise((10, 50), p=1.0),
        Resize(height=INPUT_SHAPE[0], width=INPUT_SHAPE[1], always_apply=True)
    ], p=p)


def load_train_data_images(number_of_images, shape):
    X_train = np.random.randint(0, 255, size=(number_of_images,) + shape, dtype=np.uint8)
    Y_train = np.random.randint(0, 1, size=(number_of_images,), dtype=np.uint8)
    return X_train, Y_train


def process_single_item(img):
    global GLOBAL_AUG
    img = GLOBAL_AUG(image=img)['image']
    return img


def batch_generator_train_cpu(X_train, Y_train, batch_size, prep_input):
    global INPUT_SHAPE, THREADS
    p = ThreadPool(THREADS)

    while True:
        batch_indexes = np.random.choice(list(range(X_train.shape[0])), batch_size)
        batch_image_files = X_train[batch_indexes].copy()
        batch_classes = Y_train[batch_indexes].copy()
        batch_images = p.map(process_single_item, batch_image_files)
        batch_images = np.array(batch_images, np.float32)
        if prep_input is not None:
            batch_images = prep_input(batch_images)
        yield batch_images, batch_classes


def train_mobile_net_cpu_augm():
    global BATCH_SIZE, MOBILENET_ALFA, INPUT_SHAPE, INPUT_IMAGES_SHAPE
    from tensorflow.keras.callbacks import CSVLogger
    batch_size = BATCH_SIZE
    nb_epoch = 10
    optimizer = 'Adam'
    learning_rate = 0.001

    print('Train MobileNet: Input size: {}'.format(INPUT_SHAPE))
    print('Train for {} epochs. Batch size: {}. Optimizer: {} Learing rate: {}'.format(nb_epoch, batch_size, optimizer, learning_rate))

    X_train, Y_train = load_train_data_images(100*BATCH_SIZE, INPUT_IMAGES_SHAPE)
    print('Train shape: {}'.format(X_train.shape))

    alpha = MOBILENET_ALFA
    base_model = MobileNet(INPUT_SHAPE, depth_multiplier=1, alpha=alpha, include_top=False, pooling='avg', weights='imagenet')
    x = base_model.output
    x = Dense(1, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=x)
    # print(model.summary())

    if optimizer == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])

    print('Learning rate: {}'.format(K.get_value(model.optimizer.lr)))
    history_path = os.path.join('weights_mobilenet_{:.2f}_{}px_cpu.csv'.format(alpha, INPUT_SHAPE))
    callbacks = [
        CSVLogger('history_lr_{}_optim_{}_cpu.csv'.format(learning_rate, optimizer), append=True),
    ]

    history = model.fit_generator(generator=batch_generator_train_cpu(X_train, Y_train, batch_size, preprocess_input),
                                  epochs=nb_epoch,
                                  steps_per_epoch=50,
                                  verbose=1,
                                  max_queue_size=MAX_QUEUE_SIZE,
                                  callbacks=callbacks)
    pd.DataFrame(history.history).to_csv(history_path, index=False)


if __name__ == '__main__':
    BATCH_SIZE = 64
    INPUT_IMAGES_SHAPE = (512, 512, 3)
    INPUT_SHAPE = (224, 224, 3)
    MOBILENET_ALFA = 1.0
    THREADS = 4
    MAX_QUEUE_SIZE = 10

    # GLOBAL_AUG = filp_aug(p=1.0)
    # GLOBAL_AUG = jpeg_compression_aug(p=1.0)
    # GLOBAL_AUG = color_noise_aug(p=1.0)
    GLOBAL_AUG = many_aug(p=1.0)
    train_mobile_net_cpu_augm()

