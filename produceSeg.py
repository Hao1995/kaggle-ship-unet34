import numpy as np
import pandas as pd
import os

from PIL import Image

maskDir = 'data/mask'
try:
    os.makedirs(maskDir)
except:
    print('Make directory', maskDir, 'happened error.')

masks = pd.read_csv('data/train_ship_segmentations_v2.csv', keep_default_na=False)
num_masks = masks.shape[0]
print('Total masks to encode/decode =', num_masks)

count = 0
for r in masks.itertuples():
    
    size = (768, 768)
    mask = rle_decode(r[2], size)
    
    file_path = maskDir + '/'+ r[1]

    if not r[2] == '':
        file_path = maskDir + '/'+ r[1]
        if os.path.exists(file_path):
            img = Image.open(file_path).convert('1')

            # img.show()
            pixels = img.load()
            
            for i in range(img.size[0]):
                for j in range(img.size[1]):
                    if pixels[i, j] == 0:
                        pixels[i, j] = int(mask[i][j])
                    
        else:
            # === Image Show ===
            img = Image.new('1', size)
            pixels = img.load()

            for i in range(img.size[0]):
                for j in range(img.size[1]): 
                    pixels[i, j] = int(mask[i][j])
            # img.show()
            count += 1

        img.save(file_path)
        # print('Image save ', file_path)

    # if count >= 10:
    #     break