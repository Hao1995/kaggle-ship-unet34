import os
import glob

import cv2
import numpy as np

from train_unet34_param import img_size_ori, img_size_target, SEG_PATH, TEST_IMG_PATH, SEG_RESULT_PATH
import train_unet34_model as unet34_model

model = unet34_model.UResNet34(input_shape=(img_size_ori,img_size_ori,3))

# model.load_weights("training_callback/20190107095912/weights-epoch-0002-acc-0.74.hdf5")
model.load_weights("results/20180108_train_unet34_epochs20/model_epoch20.h5")

images = glob.glob(TEST_IMG_PATH + "*.jpg") + glob.glob(TEST_IMG_PATH + "*.png") + glob.glob(TEST_IMG_PATH + "*.jpeg")
images.sort()

from skimage import img_as_ubyte

for imgName in images:
    imgName = imgName.replace('\\', '/')
    outName = imgName.replace(TEST_IMG_PATH, SEG_RESULT_PATH)

    # Input
    input = unet34_model.getImgArr(imgName, img_size_ori, img_size_ori)
    cv2.imshow('Input', input)

    # Ground Truth
    labelName = imgName.replace(TEST_IMG_PATH, SEG_PATH)
    label = unet34_model.getImgArr(labelName, img_size_ori, img_size_ori)
    cv2.imshow('Ground Truth', label)

    # Predict
    pr = model.predict(np.array([input]))[0]

    pr_rgb = cv2.cvtColor(pr,cv2.COLOR_GRAY2RGB) # channel 3
    pr_rgb_max = np.amax(pr_rgb)
    pr_rgb_tmp = pr_rgb / pr_rgb_max
    pr_rgb_tmp = pr_rgb * 255

    img_bw = np.zeros([img_size_ori,img_size_ori,3],dtype=np.uint8)
    img_bw[:,:,0] = ((pr_rgb_tmp[:,:,0] > 127) * 255).astype(np.uint8)
    img_bw[:,:,1] = ((pr_rgb_tmp[:,:,1] > 127) * 255).astype(np.uint8)
    img_bw[:,:,2] = ((pr_rgb_tmp[:,:,2] > 127) * 255).astype(np.uint8)
    cv2.imshow('Predict(black & white)', img_bw)
    cv2.imwrite(outName, img_bw)    

    # img = img_as_ubyte(pr_rgb)
    # cv2.imshow('Predict', img)
    # cv2.imwrite(outName, img)

    #

    print(os.path.basename(imgName))
    

