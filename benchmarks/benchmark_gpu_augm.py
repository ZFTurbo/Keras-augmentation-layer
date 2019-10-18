# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'

# Train MobileNet with batch generator and GPU augmentations

import os

if __name__ == '__main__':
    # Block to choose backend
    gpu_use = 0
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)
    print('GPU use: {}'.format(gpu_use))


from keras.optimizers import SGD, Adam
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.layers.core import Dense
from keras.models import Model
import pandas as pd
import numpy as np
from keras_augm_layer import *


def load_train_data_images(number_of_images, shape):
    X_train = np.random.randint(0, 255, size=(number_of_images,) + shape, dtype=np.uint8)
    Y_train = np.random.randint(0, 1, size=(number_of_images,), dtype=np.uint8)
    return X_train, Y_train


def batch_generator_train_gpu(X_train, Y_train, batch_size):
    while True:
        batch_indexes = np.random.choice(X_train.shape[0], batch_size)
        batch_images = X_train[batch_indexes].copy()
        batch_classes = Y_train[batch_indexes].copy()
        batch_images = np.array(batch_images, np.float32)
        yield batch_images, batch_classes


def get_simple_transforms():
    transforms = [
        HorizontalFlip(p=1.0),
        VerticalFlip(p=1.0),
        RandomRotate(angle=45, p=1.0),
    ]
    return transforms


def get_jpeg_compression_transforms():
    transforms = [
        JpegCompression(p=1.0, quality_lower=5, quality_upper=99),
    ]
    return transforms


def get_color_noise_transforms():
    transforms = [
        RandomBrightness(p=1.0, max_delta=0.1),
        RGBShift(p=1.0, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
        RandomContrast(0.5, 1.5, p=1.0),
        RandomHue(0.5, p=1.0),
        RandomSaturation(0.5, 1.5, p=1.0),
        RandomGaussNoise((10, 50), p=1.0),
    ]
    return transforms


def get_many_transforms():
    transforms = [
        HorizontalFlip(p=1.0),
        VerticalFlip(p=1.0),
        RandomRotate90(p=1.0),
        RandomRotate(angle=45, p=1.0),
        RandomBrightness(p=1.0, max_delta=0.1),
        RGBShift(p=1.0, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
        ToGray(p=1.0),
        JpegCompression(p=1.0, quality_lower=5, quality_upper=99),
        RandomContrast(0.5, 1.5, p=1.0),
        RandomHue(0.5, p=1.0),
        RandomSaturation(0.5, 1.5, p=1.0),
        RandomGaussNoise((10, 50), p=1.0),
    ]
    return transforms


def get_augm_model(base_model, transforms):
    global BATCH_SIZE, MOBILENET_ALFA
    from keras.layers import Input
    from keras.models import Model
    from keras.applications.mobilenet import preprocess_input

    inp = Input((None, None, 3))
    x = AugmLayer(transforms, output_dim=INPUT_SHAPE, preproc_input=preprocess_input)(inp, training=True)
    x = base_model(x)
    augm_model = Model(inputs=inp, outputs=x)
    return augm_model


def train_mobile_net_gpu_augm(transforms):
    global BATCH_SIZE, MOBILENET_ALFA, INPUT_IMAGES_SHAPE
    from keras.callbacks import CSVLogger
    batch_size = BATCH_SIZE
    nb_epoch = 10
    optimizer = 'Adam'
    learning_rate = 0.001

    print('Train MobileNet Input size: {}'.format(INPUT_SHAPE))
    print('Train for {} epochs. Batch size: {}. Optimizer: {} Learing rate: {}'.
          format(nb_epoch, batch_size, optimizer, learning_rate))

    X_train, Y_train = load_train_data_images(100 * BATCH_SIZE, INPUT_IMAGES_SHAPE)
    print('Train shape: {}'.format(X_train.shape))

    alpha = MOBILENET_ALFA
    base_model = MobileNet(INPUT_SHAPE, depth_multiplier=1, alpha=alpha, include_top=False, pooling='avg',
                           weights='imagenet')
    x = base_model.output
    x = Dense(1, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=base_model.inputs, outputs=x)
    # print(model.summary())

    model = get_augm_model(model, transforms)

    if optimizer == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])

    print('Learning rate: {}'.format(K.get_value(model.optimizer.lr)))
    history_path = os.path.join('weights_mobilenet_{:.2f}_{}px_gpu.csv'.format(alpha, INPUT_SHAPE))
    callbacks = [
        CSVLogger('history_lr_{}_optim_{}_gpu.csv'.format(learning_rate, optimizer), append=True),
    ]

    history = model.fit_generator(generator=batch_generator_train_gpu(X_train, Y_train, batch_size),
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
    MAX_QUEUE_SIZE = 10

    # transforms = get_simple_transforms()
    # transforms = get_jpeg_compression_transforms()
    # transforms = get_color_noise_transforms()
    transforms = get_many_transforms()

    train_mobile_net_gpu_augm(transforms)
