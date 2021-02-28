# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS, https://kaggle.com/zfturbo'

import tensorflow as tf
from keras.layers import Layer
from keras import backend as K


def to_tuple(param, low=None):
    if isinstance(param, (list, tuple)):
        return tuple(param)
    elif param is not None:
        if low is None:
            return -param, param
        return (low, param) if low < param else (param, low)
    else:
        return param


class HorizontalFlip():
    """Flip the input horizontally around the y-axis.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    """

    def __init__(self, p=0.5):
        self.p = tf.constant(p, dtype=tf.float32)

    def __call__(self, img, **kwargs):
        batch_size = tf.shape(img)[0]
        flips = tf.less(tf.random.uniform([batch_size], 0, 1.0, dtype=tf.float32), self.p)
        flips = tf.reshape(flips, [batch_size, 1, 1, 1])
        flips = tf.cast(flips, img.dtype)
        flipped_input = tf.reverse(img, [2])
        im = flips * flipped_input + (1 - flips) * img
        return im


class VerticalFlip():
    """Flip the input vertically around the x-axis.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    """

    def __init__(self, p=0.5):
        self.p = tf.constant(p, dtype=tf.float32)

    def __call__(self, img, **kwargs):
        batch_size = tf.shape(img)[0]
        flips = tf.less(tf.random.uniform([batch_size], 0, 1.0), self.p)
        flips = tf.reshape(flips, [batch_size, 1, 1, 1])
        flips = tf.cast(flips, img.dtype)
        flipped_input = tf.reverse(img, [1])
        im = flips * flipped_input + (1 - flips) * img
        return im


class RandomRotate():
    """Random rotate image on arbitrary angle
    Args:
        p (float): probability of applying the transform. Default: 0.5.
        angle (float): angle in grad. Default: 0.0.
    Note:
        Need to be rewritten for TF 2.0 support
    """

    def __init__(self, angle=0, p=0.5):
        self.angle = angle
        self.p = p

    def __call__(self, img, **kwargs):
        shp = tf.shape(img)
        batch_size, height, width = shp[0], shp[1], shp[2]
        coin = tf.less(tf.random.uniform([batch_size], 0, 1.0), self.p)
        angle_rad = self.angle * 3.141592653589793 / 180.0
        angles = tf.random.uniform([batch_size], -angle_rad, angle_rad)
        angles *= tf.cast(coin, tf.float32)
        f = tfa.image.angles_to_projective_transforms(angles, tf.cast(height, tf.float32), tf.cast(width, tf.float32))
        augm_img = tfa.image.transform(img, f, interpolation='BILINEAR')
        return augm_img


class RandomRotate90():
    """Randomly rotate the input by 90 degrees zero or more times.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, **kwargs):
        batch_size = tf.shape(img)[0]
        coin = tf.less(tf.random.uniform([batch_size], 0, 1.0, dtype=tf.float32), self.p)
        coin = tf.reshape(coin, [batch_size, 1, 1, 1])
        coin = tf.cast(coin, img.dtype)

        rotation_choice = tf.random.uniform([batch_size], minval=0, maxval=4, dtype=tf.int32)
        rotation_choice = tf.one_hot(rotation_choice, 4)
        flip90 = tf.image.rot90(img, k=1)
        flip180 = tf.image.rot90(img, k=2)
        flip270 = tf.image.rot90(img, k=3)

        r0 = tf.cast(tf.reshape(rotation_choice[:, 0], [batch_size, 1, 1, 1]), img.dtype)
        r1 = tf.cast(tf.reshape(rotation_choice[:, 1], [batch_size, 1, 1, 1]), img.dtype)
        r2 = tf.cast(tf.reshape(rotation_choice[:, 2], [batch_size, 1, 1, 1]), img.dtype)
        r3 = tf.cast(tf.reshape(rotation_choice[:, 3], [batch_size, 1, 1, 1]), img.dtype)

        augm_input = (r0 * img
                      + r1 * flip90
                      + r2 * flip180
                      + r3 * flip270)
        ret = coin * augm_input + (1 - coin) * img
        return ret


class RandomBrightness():
    """Adjust the brightness of images by a random factor.
    Args:
        max_delta (float): float, must be non-negative.
        p (float): probability of applying the transform. Default: 0.5.
    """

    def __init__(self, max_delta, p=0.5):
        self.p = p
        self.max_delta = max_delta

    def __call__(self, img, **kwargs):
        def random_brightness_single_image(img1):
            coin = tf.less(tf.random.uniform([1], 0, 1.0), self.p)[0]

            def apply_augm(img1):
                delta = tf.random.uniform([1], minval=-self.max_delta, maxval=self.max_delta, dtype=tf.float32)[0]
                img2 = tf.image.adjust_brightness(tf.cast(img1, tf.uint8), delta)
                return tf.cast(img2, tf.float32)

            img2 = tf.cond(coin, lambda: apply_augm(img1), lambda: img1)
            return img2

        augm_img = tf.map_fn(random_brightness_single_image, img)
        return augm_img


class RandomContrast():
    """Adjust the contrast of images by a random factor.
    Args:
        lower (float): lower bound
        upper (float): upper bound
        p (float): probability of applying the transform. Default: 0.5.
    """

    def __init__(self, lower=0.8, upper=1.2, p=0.5):
        self.p = p
        self.lower = lower
        self.upper = upper

    def __call__(self, img, **kwargs):
        def random_contrast_single_image(img1):
            coin = tf.less(tf.random.uniform([1], 0, 1.0), self.p)[0]

            def apply_augm(img1):
                img2 = tf.image.random_contrast(tf.cast(img1, tf.uint8), self.lower, self.upper)
                return tf.cast(img2, tf.float32)

            img2 = tf.cond(coin, lambda: apply_augm(img1), lambda: img1)
            return img2

        augm_img = tf.map_fn(random_contrast_single_image, img)
        return augm_img


class RGBShift():
    """Randomly shift values for each channel of the input RGB image.
    Args:
        r_shift_limit ((int, int) or int): range for changing values for the red channel. If r_shift_limit is a single
            int, the range will be (-r_shift_limit, r_shift_limit). Default: 20.
        g_shift_limit ((int, int) or int): range for changing values for the green channel. If g_shift_limit is a
            single int, the range  will be (-g_shift_limit, g_shift_limit). Default: 20.
        b_shift_limit ((int, int) or int): range for changing values for the blue channel. If b_shift_limit is a single
            int, the range will be (-b_shift_limit, b_shift_limit). Default: 20.
        p (float): probability of applying the transform. Default: 0.5.
    """

    def __init__(self, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5):
        super(RGBShift, self).__init__()
        self.p = p
        self.r_shift_limit = to_tuple(r_shift_limit)
        self.g_shift_limit = to_tuple(g_shift_limit)
        self.b_shift_limit = to_tuple(b_shift_limit)

    def __call__(self, img, **kwargs):
        def random_rgbshift_single_image(img1):
            coin = tf.less(tf.random.uniform([1], 0, 1.0), self.p)[0]

            def apply_augm(img1):
                r_shift = tf.random.uniform([1], minval=self.r_shift_limit[0], maxval=self.r_shift_limit[1], dtype=tf.float32)
                g_shift = tf.random.uniform([1], minval=self.g_shift_limit[0], maxval=self.g_shift_limit[1], dtype=tf.float32)
                b_shift = tf.random.uniform([1], minval=self.b_shift_limit[0], maxval=self.b_shift_limit[1], dtype=tf.float32)
                var = tf.stack([r_shift[0], g_shift[0], b_shift[0]])
                img2 = tf.add(img1, var)
                return img2

            img2 = tf.cond(coin, lambda: apply_augm(img1), lambda: img1)
            return img2

        augm_img = tf.map_fn(random_rgbshift_single_image, img)
        augm_img = tf.clip_by_value(augm_img, 0, 255)
        return augm_img


class ToGray():
    """Convert the input RGB image to grayscale.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    """

    def __init__(self, p=0.5):
        super(ToGray, self).__init__()
        self.p = p

    def __call__(self, img, **kwargs):
        def random_gray_image(img1):
            coin = tf.less(tf.random.uniform([1], 0, 1.0), self.p)[0]

            def apply_augm(img1):
                img2 = tf.image.rgb_to_grayscale(img1)
                img2 = tf.concat([img2, img2, img2], axis=-1)
                return img2

            img2 = tf.cond(coin, lambda: apply_augm(img1), lambda: img1)
            return img2

        augm_img = tf.map_fn(random_gray_image, img)
        return augm_img


class JpegCompression():
    """Random Jpeg compression of an image.
    Args:
        quality_lower (float): lower bound on the jpeg quality. Should be in [0, 100] range
        quality_upper (float): lower bound on the jpeg quality. Should be in [0, 100] range
    """

    def __init__(self, quality_lower=75, quality_upper=99, p=0.5):
        super(JpegCompression, self).__init__()
        self.p = p
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper

    def __call__(self, img, **kwargs):
        def random_jpeg_compress_image(img1):
            coin = tf.less(tf.random.uniform([1], 0, 1.0), self.p)[0]

            def apply_augm(img1):
                img2 = tf.image.random_jpeg_quality(tf.cast(img1, tf.uint8), self.quality_lower, self.quality_upper)
                return tf.cast(img2, tf.float32)

            img2 = tf.cond(coin, lambda: apply_augm(img1), lambda: img1)
            return img2

        augm_img = tf.map_fn(random_jpeg_compress_image, img)
        return augm_img


class RandomCrop():
    """Random crop of image
    Args:
        width_crop (tuple): (lower part of image, upper part of image) from 0 to 1.0
        height_crop (tuple): (lower part of image, upper part of image) from 0 to 1.0
        p (float): probability of applying the transform. Default: 0.5.
    """

    def __init__(self, width_crop=(0.9, 1.0), height_crop=(0.9, 1.0), p=0.5):
        self.p = p
        self.width_crop = width_crop
        self.height_crop = height_crop
        self.hc0 = tf.Variable(self.height_crop[0], dtype=tf.float32)
        self.hc1 = tf.Variable(self.height_crop[1], dtype=tf.float32)
        self.wc0 = tf.Variable(self.width_crop[0], dtype=tf.float32)
        self.wc1 = tf.Variable(self.width_crop[1], dtype=tf.float32)

    def __call__(self, img, **kwargs):
        def random_crop_single_image(img1):
            coin = tf.less(tf.random.uniform([1], 0, 1.0), self.p)[0]

            def apply_augm(img1):
                shp = tf.shape(img1)
                height, width, ch = shp[0], shp[1], shp[2]

                min_height_size = tf.cast(tf.cast(height, dtype=tf.float32) * self.hc0, tf.int32)
                max_height_size = tf.cast(tf.cast(height, dtype=tf.float32) * self.hc1, tf.int32)
                min_width_size = tf.cast(tf.cast(width, dtype=tf.float32) * self.wc0, tf.int32)
                max_width_size = tf.cast(tf.cast(width, dtype=tf.float32) * self.wc1, tf.int32)
                crop_height = tf.random.uniform([1], min_height_size, max_height_size, dtype=tf.int32)[0]
                crop_width = tf.random.uniform([1], min_width_size, max_width_size, dtype=tf.int32)[0]
                var = tf.stack([crop_height, crop_width, ch])
                img2 = tf.random_crop(img1, var)
                img2 = tf.image.resize(img2, size=(height, width))
                return img2

            img2 = tf.cond(coin, lambda: apply_augm(img1), lambda: img1)
            return img2

        augm_img = tf.map_fn(random_crop_single_image, img)
        return augm_img


class RandomHue():
    """Adjust the hue of images by a random factor.
    Args:
        max_delta (float): [0; 0.5]. Delta randomly picked in the interval [-max_delta, max_delta].
        p (float): probability of applying the transform. Default: 0.5.
    """

    def __init__(self, max_delta=0.1, p=0.5):
        self.p = p
        self.max_delta = max_delta

    def __call__(self, img, **kwargs):
        def random_hue_single_image(img1):
            coin = tf.less(tf.random.uniform([1], 0, 1.0), self.p)[0]

            def apply_augm(img1):
                img2 = tf.image.random_hue(tf.cast(img1, tf.uint8), self.max_delta)
                return tf.cast(img2, tf.float32)

            img2 = tf.cond(coin, lambda: apply_augm(img1), lambda: img1)
            return img2

        augm_img = tf.map_fn(random_hue_single_image, img)
        return augm_img


class RandomSaturation():
    """Adjust the saturation of images by a random factor.
    Args:
        lower (float): lower bound. Default: 0.8
        upper (float): upper bound. Default: 1.2
        p (float): probability of applying the transform. Default: 0.5.
    """

    def __init__(self, lower=0.8, upper=1.2, p=0.5):
        self.p = p
        self.lower = lower
        self.upper = upper

    def __call__(self, img, **kwargs):
        def random_saturation_single_image(img1):
            coin = tf.less(tf.random.uniform([1], 0, 1.0), self.p)[0]

            def apply_augm(img1):
                img2 = tf.image.random_saturation(tf.cast(img1, tf.uint8), self.lower, self.upper)
                return tf.cast(img2, tf.float32)

            img2 = tf.cond(coin, lambda: apply_augm(img1), lambda: img1)
            return img2

        augm_img = tf.map_fn(random_saturation_single_image, img)
        return augm_img


class RandomGaussNoise():
    """Apply gaussian noise to the input image.
    Args:
        var_limit ((int, int) or int): variance range for noise. If var_limit is a single int, the range
            will be (-var_limit, var_limit). Default: (10, 50).
        p (float): probability of applying the transform. Default: 0.5.
    """

    def __init__(self, var_limit=(10, 50), p=0.5):
        self.p = p
        self.var_limit = to_tuple(var_limit)
        self.vl0 = tf.Variable(self.var_limit[0], dtype=tf.float32)
        self.vl1 = tf.Variable(self.var_limit[1], dtype=tf.float32)

    def __call__(self, img, **kwargs):
        def random_gauss_noise_single_image(img1):
            coin = tf.less(tf.random.uniform([1], 0, 1.0), self.p)[0]

            def apply_augm(img1):
                noise_mean = tf.random.uniform([1], self.vl0, self.vl1, dtype=tf.float32)[0]
                noise_std = noise_mean ** 0.5
                noise = tf.random.normal(shape=tf.shape(img1), mean=noise_mean, stddev=noise_std, dtype=tf.float32)
                noise -= tf.reduce_min(noise)
                img2 = img1 + noise
                return img2

            img2 = tf.cond(coin, lambda: apply_augm(img1), lambda: img1)
            return img2

        augm_img = tf.map_fn(random_gauss_noise_single_image, img)
        augm_img = tf.clip_by_value(augm_img, 0, 255)
        return augm_img


class ResizeImage():
    """Resize image.
    Args:
        width (int): width
        height (int): height
    """

    def __init__(self, width, height):
        self.width = tf.Variable(width, dtype=tf.int32)
        self.height = tf.Variable(height, dtype=tf.int32)
        self.size = tf.constant((height, width), dtype=tf.int32)

    def __call__(self, img, **kwargs):
        shp = tf.shape(img)
        batch_size, h, w, ch = shp[0], shp[1], shp[2], shp[3]
        if h != self.height or w != self.width:
            img = tf.image.resize(img, size=self.size)
        return img


class ReadImage():
    """Read and decode image with tensorflow.
    """

    def __init__(self,):
        super(ReadImage, self).__init__()

    def __call__(self, img, **kwargs):
        def read_single_image(img_path):
            img1 = tf.io.decode_image(tf.read_file(img_path[0][0]))
            return tf.cast(img1, dtype=tf.float32)

        augm_img = tf.map_fn(read_single_image, img)
        return tf.cast(augm_img, dtype=tf.float32)


def augment(inputs, transforms):
    for t in transforms:
        inputs = t(inputs)

    return inputs


def resize_if_needed(img, output_dim):
    shp = tf.shape(img)
    batch_size, height, width, channels = shp[0], shp[1], shp[2], shp[3]
    if height != output_dim[0] or width != output_dim[1]:
        size = tf.constant(output_dim[:2], dtype=tf.int32)
        img = tf.image.resize(img, size=size)
    return img


class AugmLayer(Layer):
    def __init__(self, transforms, output_dim=None, preproc_input=None, **kwargs):
        self.output_dim = output_dim
        self.transforms = transforms
        self.preproc_input = preproc_input
        super(AugmLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AugmLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        ret = K.in_train_phase(augment(inputs, self.transforms), inputs, training=training)
        if self.output_dim is not None:
            ret = resize_if_needed(ret, self.output_dim)
        if self.preproc_input is not None:
            ret = self.preproc_input(ret)
        return ret

    def compute_output_shape(self, input_shape):
        if self.output_dim is None:
            return input_shape
        else:
            return (None, ) + self.output_dim
