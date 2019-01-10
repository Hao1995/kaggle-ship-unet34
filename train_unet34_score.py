import os
import numpy as np
from scipy.ndimage import label

from train_unet34_param import threshold, img_size_ori

# 把一張圖的每個mask分開並回傳
def split_mask(img):
    assert img.shape[2] == 1
    labled, n_objs = label(img)
    result = []
    for i in range(n_objs):
        obj = (labled == i + 1).astype(int)
        result.append(obj)
    return result

def get_ground_truth(img_id, df, img_size_ori): #return mask for each ship
    masks = df.loc[img_id]['EncodedPixels']
    if(type(masks) == float): return []
    if(type(masks) == str): masks = [masks]
    result = []
    for mask in masks:
        img = np.zeros(img_size_ori*img_size_ori, dtype=np.uint8)
        s = mask.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1
        img = img.reshape((img_size_ori, img_size_ori)).T
        img = np.expand_dims(img, axis=2)
        result.append(img)
    return result

def IoU(pred, ground):
    inter = (pred*ground).sum()
    return inter / ((pred+ground).sum() - inter)

def get_score(preds, grounds, threshold):
    b = 4 # B^2 , B=2 >> b=4
    n_th = 10
    thresholds = [threshold + 0.05*i for i in range(n_th)]
    score = 0

    for t in thresholds:
        # tp : 結果是true,預測是true
        # tn : 結果是false,預測是false,
        # fp : 結果是false,預測是true
        # fn : 結果是true,預測是false

        tp, fp, fn = 0, 0, 0
        tp_map_dict = {}
        preds_cp = preds.copy()

        for ground_idx, ground in enumerate(grounds):
            h_idx = -1
            h_score = t
            for pred_idx, pred in enumerate(preds_cp): 
                iou_score = IoU(pred, ground)
                if iou_score >= t or iou_score > h_score:
                    h_idx = pred_idx
                    h_score = iou_score

            if h_idx > -1:
                tp_map_dict[ground_idx] = [h_idx, h_score]
            else:
                fn += 1
            del preds_cp[h_idx]
                
        tp = len(tp_map_dict)
        fp = len(preds_cp)

        score += ((b+1)*tp)/((b+1)*tp + b*fn + fp)       
    return score/n_th

# imgs = ...
# preds = split_mask(imgs)

# seg_df = ...
# grounds = get_ground_true(img_id, seg_df, img_size_ori)


# === Calculate Score Test ===
# ground_img = np.array([[0,1,1,1,0],
#                         [0,0,0,0,0],
#                         [1,1,0,1,1],
#                         [0,1,1,0,0],
#                         [0,0,1,1,0]])
# ground_img = np.expand_dims(ground_img, axis=2)

# pred_img = np.array([[0,1,1,0,0],
#                     [0,0,0,0,0],
#                     [1,1,0,1,1],
#                     [0,1,1,0,0],
#                     [0,0,1,0,0]])
# pred_img = np.expand_dims(pred_img, axis=2)

# grounds = split_mask(ground_img)
# preds = split_mask(pred_img)

# score = get_score(preds, grounds, threshold)

# print(score)
# =============================