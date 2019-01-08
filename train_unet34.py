import numpy as np
import pandas as pd
import cv2
import os
import itertools

from keras import backend as K
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization

from skimage.transform import resize

# import train_unet34_param as unet34Param
from train_unet34_param import IMG_PATH, SEG_PATH, SAVE_WEIGHTS_PATH, img_size_ori, img_size_target, batch_size, epochs

# === Up & Down Sample ===

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
# imgs_name = [f for f in os.listdir(IMG_PATH) if os.path.isfile(os.path.join(IMG_PATH, f))]

# train_img = [np.array(load_img(IMG_PATH + "/{}".format(idx))) / 255 for idx in imgs_name] # load all images at once

def getImgArr(path, width, height, imgNorm="none"):
    # print('getImgArr')

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
    # print('getSegArr')
    # seg_labels = np.zeros((height, width, 2))
    try:
        img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_gray = cv2.resize(img_gray, (width, height))
        # cv2.imshow('seg', img_gray)

        (thresh, img_bool) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # cv2.imshow('seg_reshpe', img_bool)
        img = np.expand_dims(img_bool, axis=2)
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

# === Check length of input image is same with length of seg images ===
TRAIN_IMG_NUM = len([name for name in os.listdir(IMG_PATH) if os.path.isfile(os.path.join(IMG_PATH, name))])
TRAIN_SEG_NUM = len([name for name in os.listdir(SEG_PATH) if os.path.isfile(os.path.join(SEG_PATH, name))])
if TRAIN_IMG_NUM != TRAIN_SEG_NUM:
    raise Exception('TRAIN_IMG_NUM = ', TRAIN_IMG_NUM , ' ; TRAIN_SEG_NUM = ', TRAIN_SEG_NUM, '. Is not equal.')

gen = imgGenerator(IMG_PATH, SEG_PATH, batch_size, img_size_ori, img_size_target)

# === Loss Function ===
from keras.losses import binary_crossentropy
from keras import backend as K

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# === Model Fitting ===
import train_unet34_model as unet34_model

model = unet34_model.UResNet34(input_shape=(img_size_target,img_size_target,3))
model.summary()

# === 保存和恢復模型 ===
import os
from datetime import datetime

checkpoint_path = "training_callback" + "/" + datetime.now().strftime('%Y%m%d%H%M%S')
try:
    os.makedirs(checkpoint_path)
except:
    print('Make directory', checkpoint_path, 'happend error.')

checkpoint_file = checkpoint_path + "/weights-epoch-{epoch:04d}-acc-{acc:.3f}.hdf5"
model_checkpoint = ModelCheckpoint(checkpoint_file, monitor='acc',save_weights_only=True, verbose=1)
model.compile(loss=bce_dice_loss, optimizer="adam", metrics=["accuracy"])

history = model.fit_generator(gen,
                    TRAIN_IMG_NUM // batch_size,
                    epochs=epochs,
                    callbacks=[model_checkpoint])

model.save_weights(SAVE_WEIGHTS_PATH + "model_epochs." + str(epochs))
model.save(SAVE_WEIGHTS_PATH + "model_epoch" + str(epochs) + ".h5")