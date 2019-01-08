import os
import glob

import cv2
import numpy as np

from keras.losses import binary_crossentropy
from keras import backend as K

# import train_unet34_param as unet34Param
from train_unet34_param import img_size_ori, img_size_target, TEST_IMG_PATH, SEG_RESULT_PATH

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

# === Loss Function ===

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

model = unet34_model.UResNet34(input_shape=(img_size_ori,img_size_ori,3))

# model.load_weights("training_callback/20190107095912/weights-epoch-0002-acc-0.74.hdf5")
model.load_weights("results/model_epoch20.h5")

model.compile(loss=bce_dice_loss, optimizer="adam", metrics=["accuracy"])

images = glob.glob(TEST_IMG_PATH + "*.jpg") + glob.glob(TEST_IMG_PATH + "*.png") + glob.glob(TEST_IMG_PATH + "*.jpeg")
images.sort()

colors = [(0, 0, 0), (255, 255, 255)]

for imgName in images:
    imgName = imgName.replace('\\', '/')
    outName = imgName.replace(TEST_IMG_PATH, SEG_RESULT_PATH)
    X = getImgArr(imgName, img_size_ori, img_size_ori)
    pr = model.predict(np.array([X]))[0]
    # pr = pr.reshape((img_size_target, img_size_target, n_classes)).argmax(axis=2)
    # seg_img = np.zeros((img_size_target, img_size_target, 3))
    # for c in range(n_classes):
    #     seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
    #     seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
    #     seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
    # seg_img = cv2.resize(seg_img, (input_width, input_height))
    cv2.imwrite(outName, pr)
    cv2.imshow(outName, pr)
