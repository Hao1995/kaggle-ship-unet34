import numpy as np
import os

from keras import backend as K
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization

from skimage.transform import resize

from train_unet34_param import IMG_PATH, SEG_PATH, SAVE_WEIGHTS_PATH, img_size_ori, img_size_target, batch_size, epochs
import train_unet34_model as unet34_model

# === Check length of input image is same with length of seg images ===
TRAIN_IMG_NUM = len([name for name in os.listdir(IMG_PATH) if os.path.isfile(os.path.join(IMG_PATH, name))])
TRAIN_SEG_NUM = len([name for name in os.listdir(SEG_PATH) if os.path.isfile(os.path.join(SEG_PATH, name))])
if TRAIN_IMG_NUM != TRAIN_SEG_NUM:
    raise Exception('TRAIN_IMG_NUM = ', TRAIN_IMG_NUM , ' ; TRAIN_SEG_NUM = ', TRAIN_SEG_NUM, '. Is not equal.')

gen = unet34_model.imgGenerator(IMG_PATH, SEG_PATH, batch_size, img_size_ori, img_size_target)

# === Model Fitting ===
model = unet34_model.UResNet34(input_shape=(img_size_target,img_size_target,3))
model.summary()

# === 保存和恢復模型 ===
import os
from datetime import datetime

checkpoint_path = "training_callback" + "/" + datetime.now().strftime('%Y%m%d%H%M')
try:
    os.makedirs(checkpoint_path)
except:
    print('Make directory', checkpoint_path, 'happend error.')

checkpoint_file = checkpoint_path + "/weights-epoch-{epoch:04d}-acc-{acc:.3f}.hdf5"
model_checkpoint = ModelCheckpoint(checkpoint_file, monitor='acc',save_weights_only=True, verbose=1)
model.compile(loss=unet34_model.bce_dice_loss, optimizer="adam", metrics=["accuracy"])

history = model.fit_generator(gen,
                    TRAIN_IMG_NUM // batch_size,
                    epochs=epochs,
                    callbacks=[model_checkpoint])

model.save_weights(SAVE_WEIGHTS_PATH + "model_epochs." + str(epochs))
model.save(SAVE_WEIGHTS_PATH + "model_epoch" + str(epochs) + ".h5")