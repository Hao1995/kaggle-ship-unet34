import numpy as np
import pandas as pd
import os

import os.path

from PIL import Image

def rle_encode(img):
    '''
    not yet fixed
    '''
    # pixels = img.flatten()
    # pixels = np.concatenate([[0], pixels, [0]])
    # runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # runs[1::2] -= runs[::2]
    # return ' '.join(str(x) for x in runs)
 
def rle_decode(mask_str, shape):
    s = mask_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(shape)
    
maskDir = 'data/label'

try:
    os.makedirs(maskDir)
except:
    print('Make directory', maskDir, 'happened error.')

masks = pd.read_csv('data/train_ship_segmentations_v2.csv', keep_default_na=False)
num_masks = masks.shape[0]
print('Total masks to encode/decode =', num_masks)

count = 0
for r in masks.itertuples():
    
    if os.path.isfile('data/train/' + r[1]) :

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
