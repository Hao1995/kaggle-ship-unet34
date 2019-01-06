from keras.models import Model
from keras.layers import Input, merge, core, Dropout, concatenate, ZeroPadding2D, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D


def Unet(nClasses, optimizer=None, input_width=64, input_height=96, nChannels=1):
    inputs = Input((input_height, input_width, nChannels))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    conv6 = Conv2D(nClasses, (1, 1), activation='relu', padding='same')(conv5)
    conv6 = core.Reshape((nClasses, input_height * input_width))(conv6)
    conv6 = core.Permute((2, 1))(conv6)

    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    if not optimizer is None:
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return model


def segnet(nClasses, optimizer=None, input_height=96, input_width=64):
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    inputs = Input((input_height, input_width, 1))

    # encoder
    x = ZeroPadding2D(padding=(pad, pad))(inputs)
    x = Conv2D(filter_size, (kernel, kernel), padding='valid')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)

    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Conv2D(128, (kernel, kernel), padding='valid')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)

    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Conv2D(256, (kernel, kernel), padding='valid')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)

    # decoder
    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Conv2D(256, (kernel, kernel), padding='valid')(x)

    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Conv2D(128, (kernel, kernel), padding='valid')(x)

    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Conv2D(128, (kernel, kernel), padding='valid')(x)

    x = Conv2D(nClasses, (1, 1), padding='valid')(x)

    x = core.Reshape((nClasses, input_height * input_width),)(x)

    x = core.Permute((2, 1))(x)
    x = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    if not optimizer is None:
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return model

# === Loss Function ===
import tensorflow as tf
from keras import backend as K

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def focal_loss(y_true, y_pred, alpha=10.0, gamma=2.0):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def MixedLoss(y_true, y_pred, beta=10.0, alpha=0.75, gamma=2.0):
    return beta*focal_loss(alpha, gamma, y_true, y_pred) - K.log(dice_loss(y_true, y_pred))