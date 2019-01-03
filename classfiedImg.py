import pandas as pd
import os

pictureDir = 'data/train'
masks = pd.read_csv('data/train_ship_segmentations_v2.csv', keep_default_na=False)
num_masks = masks.shape[0]
print('Total masks to encode/decode =', num_masks)

directoryShip = 'data/train/ship'
try:
    os.makedirs(directoryShip)
except:
    print('Make directory', directoryShip, 'happened error.')

directoryNoShip = 'data/train/no_ship'
try:
    os.makedirs(directoryNoShip)
except:
    print('Make directory', directoryNoShip, 'happened error.')

for r in masks.itertuples():
    
    fileName = r[1]
    if not r[2] == '':
        print(fileName, 'has ship.')
        try:
            os.rename(pictureDir + "/" + fileName, directoryShip + "/" + fileName)
        # except OSError as e:
        #     if e.errno != errno.EEXIST:
        #         raise
        except:
            print(fileName + 'does not exist in ' + pictureDir)
    else :
        print(fileName, 'has no ship.')
        try:
            os.rename(pictureDir + "/" + fileName, directoryNoShip + "/" + fileName)
        except:
            print(fileName + 'does not exist in ' + pictureDir)