Keras implementation of layer which performs augmentations of images using GPU. It can be a good and fast replacement for ImageDataGenerator or independent augmentation libraries like [imgaug](https://github.com/aleju/imgaug), [albumentations](https://github.com/albu/albumentations) etc. 

**Important Note**: It's prototype version which I believe can be improved a lot in terms of speed and usability. **I'd really like to see it in official Keras and TF repository.** Feel free to add more transforms or parameters using pull requests.

# Requirements

Python 3.\*, Keras 2.\*, tensorflow 1.14

# Usage

You need to define initial model, then choose set of transforms and add AugmLayer as first layer of your model. See example below:

```python
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.layers import Dense, Input
from keras.models import Model
from keras_augm_layer import *
    
def get_classification_model():
    base_model = MobileNet((128, 128, 3), depth_multiplier=1, alpha=0.25,
                           include_top=False, pooling='avg', weights='imagenet')
    x = base_model.layers[-1].output
    x = Dense(1, use_bias=False)(x)
    model = Model(inputs=base_model.inputs, outputs=x)
    return model
    
transforms = [
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomRotate(angle=45, p=0.5),
    RandomRotate90(p=0.5),
    
    RGBShift(p=0.1, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
    RandomBrightness(p=0.999, max_delta=0.1),
    RandomContrast(0.5, 1.5, p=0.9),
    RandomHue(0.5, p=0.9),
    RandomSaturation(0.5, 1.5, p=0.9),
    
    RandomGaussNoise((10, 50), p=0.99),
    
    ToGray(p=0.5),
    JpegCompression(p=0.9, quality_lower=5, quality_upper=99),
]

inp = Input((None, None, 3))
x = AugmLayer(transforms, output_dim=(128, 128, 3), preproc_input=preprocess_input)(inp)
base_model = get_classification_model()
x = base_model(x)
augm_model = Model(inputs=inp, outputs=x)
```

* Note 1: You need to specify output dimension of AugmLayer to be the same as model you train
* Note 2: You need to specify ```preprocess_input``` function, because it must be applied to image before sent on model input.
* Note 3: Because of Note 2 you must send images "as is" without preprocessing and even without resizing. AugmLayer will make resize automatically if necessary.  

# Debuging

Before starting training process you most likely will want to check your set of transforms. See example how to debug below:
 
```python
# Define set of transforms
transforms = [
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
]

# Create Keras Model only with single AugmLayer (don't use preprocess_input and specify training=True)
inp = Input((None, None, 3))
x = AugmLayer(transforms, output_dim=(128, 128, 3))(inp, training=True)
augm_model = Model(inputs=inp, outputs=x)

# Read and predict images
files = glob.glob("../input/*.jpg")
images = []
for f in files:
    images.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
images = np.array(images)
# augm_images will contain augmented images
augm_images = augm_model.predict(images)

# Now show all augmented images
def show_image(im, name='image'):
    cv2.imshow(name, cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for i in range(augm_images.shape[0]):
    show_image(augm_images[i])
```

You can use this script for debug: [check_augmentations.py]()

# Saving trained model

AugmLayer has the same behaviour as Dropout layer. It's applied transformations only during training phase. So you can save model as is. 
But exists the better way. If you create augm_model as it was in Usage section you can just pull out initial classification model with 
construction below:
  
```python
model = augm_model.layers[2]
model.save(...)
```

# Benchmark

- Script 1: [Benchmark CPU (Albumentations)]()
- Script 2: [Benchmark GPU (AugmLayer)]()

* GPU: 1080Ti + CPU: 6 Core Intel
* Input images shape: (512, 512, 3)
* Batch size: 64
* MobileNet_v1 (alpha=1.0)
* Input shape: (224, 224, 3)
* Images: Randomly generated (0-255 uint8)
* Probability = 1.0 for all
* Steps per epoch: 50

| Set of augm   | Albumentations (Threads 1) | Albumentations (Threads 4) | Albumentations (Threads 6) | AugmLayer |
|---------------|----------------------------|----------------------------|----------------------------|-----------|
| Flips + RandomRotate45 | 15s | 14s | 14s | 19s |
| JpegCompression | 38s | 15s | 15s | 21s |
| Colors + Noise | 147s | 68s | 62s | 32s |
| Many Augm | 176s | 76s | 69s | 41s |

Some observations: GPU augmentations became much faster than CPU in following cases:
1) You have some heavy operations or small number of weak CPU-cores
2) You need to apply large set of different augmentations for each image
e.g. you don't have enough CPU power to prepare new batch for time while GPU work on previous batch. 
It's expected that GPU augmentations will work better on smaller neural nets.  

# Limitations & drawbacks 

* Augmentations only suitable to train classification models. Current implementation does not support boxes, masks, keypoints.
* Model with augmentation layer requires more GPU memory (sometimes you will need to decrease batch_size)

# Current list of all available transforms

* HorizontalFlip
* VerticalFlip
* RandomRotate
* RandomRotate90
* RandomBrightness
* RandomContrast
* RGBShift
* ToGray
* JpegCompression
* RandomHue
* RandomSaturation
* RandomCrop
* RandomGaussNoise
* ResizeImage
* ReadImage

# ToDo

* Make it much faster than usage of CPU augmentations
* Add support of OneOf() construction for more flexible choose of transforms
* Add more different transforms like '_Blur_', '_Normalize_', '_ShiftScaleRotate_', '_CenterCrop_', '_OpticalDistortion_', 
'_GridDistortion_', '_ElasticTransform_', '_MotionBlur_', '_MedianBlur_', '_CLAHE_', '_ChannelShuffle_', '_InvertImg_', '_RandomScale_', 
'_Resize_', '_AffineTransforms_' etc
* Add support for boxes, masks and keypoints. But it will require changes in training process not only model definition. Need to think about easiest way to do it.
* Check to work with TensorFlow 2.* (currently RandomRotateAugm won't work in TF 2.0) 
