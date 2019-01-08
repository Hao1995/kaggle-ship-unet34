import os
import glob

import cv2
import numpy as np

from keras.losses import binary_crossentropy
from keras import backend as K

from train_unet34_param import img_size_ori, img_size_target, TEST_IMG_PATH, SEG_RESULT_PATH
import train_unet34_model as unet34_model

model = unet34_model.UResNet34(input_shape=(img_size_ori,img_size_ori,3))

# model.load_weights("training_callback/20190107095912/weights-epoch-0002-acc-0.74.hdf5")
model.load_weights("results/20180108_train_unet34_epochs20/model_epoch20.h5")

# model.compile(loss=bce_dice_loss, optimizer="adam", metrics=["accuracy"])

images = glob.glob(TEST_IMG_PATH + "*.jpg") + glob.glob(TEST_IMG_PATH + "*.png") + glob.glob(TEST_IMG_PATH + "*.jpeg")
images.sort()

from skimage import img_as_ubyte

for imgName in images:
    imgName = imgName.replace('\\', '/')
    outName = imgName.replace(TEST_IMG_PATH, SEG_RESULT_PATH)
    X = unet34_model.getImgArr(imgName, img_size_ori, img_size_ori)
    # cv2.imshow('X', X)

    pr = model.predict(np.array([X]))[0]

    pr_rgb = cv2.cvtColor(pr,cv2.COLOR_GRAY2RGB)
    # pr_rgb_name = outName.replace(os.path.basename(outName), 'graytorgb.jpg')
    # cv2.imshow(pr_rgb_name, pr_rgb)
    # cv2.imwrite(pr_rgb_name, pr_rgb)

    img = img_as_ubyte(pr_rgb)
    # cv2.imshow(outName, img)
    cv2.imwrite(outName, img)
