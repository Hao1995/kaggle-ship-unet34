import numpy as np
import pandas as pd
import cv2

from keras import backend as K
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization

from skimage.transform import resize

import os
import itertools

# === Up & Down Sample ===
img_size_ori = 768
img_size_target = 768
batch_size = 8 # 256:64, 384:32, 768:6(8)

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    #res = np.zeros((img_size_target, img_size_target), dtype=img.dtype)
    #res[:img_size_ori, :img_size_ori] = img
    #return res
    
def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
    #return img[:img_size_ori, :img_size_ori]

# === Load Img ===
IMG_PATH = 'data/train/'
SEG_PATH = 'data/label/'
# imgs_name = [f for f in os.listdir(IMG_PATH) if os.path.isfile(os.path.join(IMG_PATH, f))]

# train_img = [np.array(load_img(IMG_PATH + "/{}".format(idx))) / 255 for idx in imgs_name] # load all images at once

def getImgArr(path, width, height, imgNorm="none"):

    try:
        img = cv2.imread(path)
        img = cv2.resize(img, (width, height))
        # img = cv2.imread(path, cv2.IMREAD_COLOR)
        # cv2.imshow(imgNorm+'0', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # im = np.zeros((height, width, 1))
        # cv2.imshow(imgNorm+'1', im)

        if imgNorm == "sub_and_divide":
            # Preprocess Input >> mode = tf (will scale pixels between -1 and 1)
            # im[:,:,0] = np.float32(img) / 127.5 -1 
            # im[:,:,0] = np.float32(cv2.resize(img, (width, height))) / 127.5 -1 
            img = np.float32(img) / 127.5 - 1
        elif imgNorm == "sub_mean":
            # Preprocess Input >> mode = caffe (will convert the images from RGB to BGR, then will zero-center each color channel with respect to the ImageNet dataset)
            img = img.astype(np.float32)
            img[:, :, 0] -= 103.939 # B
            img[:, :, 1] -= 116.779 # G
            img[:, :, 2] -= 123.68 # R
        elif imgNorm == "divide":
            # Preprocess Input >> mode = torch (will scale pixels between 0 and 1 and then will normalize each channel with respect to the ImageNet dataset)
            img = img.astype(np.float32)
            img = img / 255.0
            
        # cv2.imshow(imgNorm+'2', img)
        return img
    except Exception as e:
        print(path, e)
        img= np.zeros((height, width, 3))
        return img

def getSegArr(path, width, height):

    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (width, height))
        img = img[:, :, 0]
        # cv2.imshow('seg', img)

        # for c in range(nClasses):
        #     seg_labels[:, :, c] = (img == c).astype(int)
        return img

    except Exception as e:
        print(e)
        img = np.zeros((height, width, 1))
        return img

def imgGenerator(imgs_path, segs_path, batch_size, input_size, output_size):

    imgs = [f for f in os.listdir(imgs_path) if os.path.isfile(os.path.join(imgs_path, f))]
    segs = [f for f in os.listdir(segs_path) if os.path.isfile(os.path.join(segs_path, f))]
    imgs.sort()
    segs.sort()

    for i in range(len(imgs)):
        imgs[i] = imgs_path + imgs[i]
        segs[i] = segs_path + segs[i]

        # img = cv2.imread(images[i], cv2.IMREAD_COLOR)
        # cv2.imshow('image-origin',img)
        # img = cv2.imread(segmentations[i], cv2.IMREAD_COLOR)
        # cv2.imshow('image-seg',img)

    assert len(imgs) == len(segs)
    for img, seg in zip(imgs, segs):
        assert (img.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])

    # 'itertools.cycle' EX: [a,b,c] >> [a,b,c,a,b,c,a, ... , c, a, ...]. infinite loop.
    zipped = itertools.cycle(zip(imgs, segs))

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = next(zipped)
            X.append(getImgArr(im, input_size, input_size))
            Y.append(getSegArr(seg, input_size, input_size))

        yield np.array(X), np.array(Y)

imgGenerator(IMG_PATH, SEG_PATH, batch_size, img_size_ori, img_size_target)
# model.fit_generator(
#         train_generator)

