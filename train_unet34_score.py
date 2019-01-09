import os
import numpy as np
from scipy.ndimage import label

from train_unet34_param import threshold

# 把一張圖的每個mask分開並回傳
def split_mask(img, threshold=0.5):
    assert img.shape[2] == 1
    labled, n_objs = label(img)
    result = []
    for i in range(n_objs):
        obj = (labled == i + 1).astype(int)
        result.append(obj)
    return result

def get_ground_true(img_id, df, shape = (768,768)): #return mask for each ship
    masks = df.loc[img_id]['EncodedPixels']
    if(type(masks) == float): return []
    if(type(masks) == str): masks = [masks]
    result = []
    for mask in masks:
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        s = mask.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1
        result.append(img.reshape(shape).T)
    return result