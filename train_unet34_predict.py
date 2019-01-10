import os
import glob

import cv2
import numpy as np
import pandas as pd

from train_unet34_param import img_size_ori, img_size_target, threshold
from train_unet34_param import SEG_PATH,SCORE_RESULT_PATH, SEG_FILE, TEST_IMG_PATH, SEG_RESULT_PATH, SCORE_RESULT_PATH
import train_unet34_score as eval_score
import train_unet34_model as unet34_model

# === Load Ground Truth File ===
print('Load Segmentation File >>>>>>')
seg_df = pd.read_csv(SEG_FILE).set_index('ImageId')
# ==============================

print('Prepare Model >>>>>>')
model = unet34_model.UResNet34(input_shape=(img_size_ori,img_size_ori,3))
# model.summary()

print('Load Weights >>>>>>')
# model.load_weights("training_callback/20190107095912/weights-epoch-0002-acc-0.74.hdf5")
model.load_weights("results/20190109_train_unet34_epochs1_inputAll/weights-epoch-0001-acc-0.887.hdf5")

images = glob.glob(TEST_IMG_PATH + "*.jpg") + glob.glob(TEST_IMG_PATH + "*.png") + glob.glob(TEST_IMG_PATH + "*.jpeg")
images.sort()

# === Make a directory for saving the score file ===
try:
    os.makedirs(SCORE_RESULT_PATH)
except:
    print('Make directory', SCORE_RESULT_PATH, 'happend error.')
# ==================================================

# === Create A Score CSV File ===
# import csv

# score_file = SCORE_RESULT_PATH + 'ship_score.csv'
# csvfile = open(score_file, 'w')
# filewriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
# filewriter.writerow(['ImageId', 'F2-Score'])
# ===============================

# from skimage import img_as_ubyte

for imgName in images:
    imgName = imgName.replace('\\', '/')
    img_id = os.path.basename(imgName)
    # outName = imgName.replace(TEST_IMG_PATH, SEG_RESULT_PATH)

    # Input
    input = unet34_model.getImgArr(imgName, img_size_ori, img_size_ori)
    # cv2.imshow('Input', input)

    # === Ground Truth ===
    # By Real Img
    # labelName = imgName.replace(TEST_IMG_PATH, SEG_PATH)
    # label = unet34_model.getImgArr(labelName, img_size_ori, img_size_ori)
    # By Seg CSV
    label = unet34_model.getSegArrByCSV(img_id, seg_df, img_size_ori, img_size_ori) # chan1, 0~255
    # cv2.imshow('Ground Truth', label)
    # ====================

    # === Pred ===
    pr = model.predict(np.array([input]))[0] # chan 1
    # cv2.imshow('pr', pr)

    # === Pred - Convert 1 Channels Image to Boolean Image ===
    pr_b = np.copy(pr)
    pr_b_max = np.amax(pr_b)
    pr_b = pr_b / pr_b_max 
    pr_b = (pr_b > 0.5).astype(np.uint8) # 0~1
    # cv2.imshow('pr_b', pr_b*255)
    # ========================================================

    # === Evaluate ===
    preds = eval_score.split_mask(pr_b)
    grounds = eval_score.get_ground_truth(img_id, seg_df, img_size_ori)
    score = eval_score.get_score(preds, grounds, threshold)
    # ================

    # === Store 3 Channels(RGB) Image Of Prediction ===
    # pr_rgb = cv2.cvtColor(pr,cv2.COLOR_GRAY2RGB) # channel 3
    # pr_rgb_max = np.amax(pr_rgb)
    # pr_rgb_tmp = pr_rgb / pr_rgb_max
    # pr_rgb_tmp = pr_rgb * 255

    # img_bw = np.zeros([img_size_ori,img_size_ori,3],dtype=np.uint8)
    # img_bw[:,:,0] = ((pr_rgb_tmp[:,:,0] > 127) * 255).astype(np.uint8)
    # img_bw[:,:,1] = ((pr_rgb_tmp[:,:,1] > 127) * 255).astype(np.uint8)
    # img_bw[:,:,2] = ((pr_rgb_tmp[:,:,2] > 127) * 255).astype(np.uint8)
    # # cv2.imshow('Predict(black & white)', img_bw)
    # cv2.imwrite(outName, img_bw)
    # ======================================

    # === Make value range from 0 to 255 ===
    # img = img_as_ubyte(pr_rgb)
    # cv2.imshow('Predict', img)
    # cv2.imwrite(outName, img)
    # ======================================

    # === Combine Pred Img & Ground Truth ===
    img_bl = np.zeros([img_size_ori,img_size_ori,3],dtype=np.uint8)
    pr_b_tmp = pr_b*255
    img_bl[:,:,2] = (label[:,:,0] - pr_b_tmp[:,:,0]).clip(min=0) #R
    img_bl[:,:,1] = (pr_b_tmp[:,:,0] - label[:,:,0]).clip(min=0) #G
    img_bl[:,:,0] = pr_b[:,:,0]*label[:,:,0] #B
    # =======================================

    # === Plot Result ===
    cv2.imshow('Input', input)
    cv2.imshow('Ground Truth', label)
    cv2.imshow('Prediction', pr_b*255)
    cv2.imshow('Intersection', img_bl)
    # ===================

    # === Write to CSV ===
    # filewriter.writerow([img_id, score])
    # ====================

    print(img_id,' - score :', score)
    
# csvfile.close()