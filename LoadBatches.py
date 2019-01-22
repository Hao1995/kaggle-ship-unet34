import numpy as np
import cv2
import glob
import random
import itertools
import os

def getImageArr(path, width, height, imgNorm="sub_and_divide"):
    """
    
    """
    try:
        # 'cv2.imread' : 0 flag means grayscale mode
        # img = cv2.imread(path, 0)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        # cv2.imshow(imgNorm+'0', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # im = np.zeros((height, width, 1))
        # cv2.imshow(imgNorm+'1', im)

        if imgNorm == "sub_and_divide":
            # Preprocess Input >> mode = tf (will scale pixels between -1 and 1)
            # im[:,:,0] = np.float32(img) / 127.5 -1 
            # im[:,:,0] = np.float32(cv2.resize(img, (width, height))) / 127.5 -1 
            img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
        elif imgNorm == "sub_mean":
            # Preprocess Input >> mode = caffe (will convert the images from RGB to BGR, then will zero-center each color channel with respect to the ImageNet dataset)
            img = cv2.resize(img, (width, height))
            img = img.astype(np.float32)
            img[:, :, 0] -= 103.939 # B
            img[:, :, 1] -= 116.779 # G
            img[:, :, 2] -= 123.68 # R
        elif imgNorm == "divide":
            # Preprocess Input >> mode = torch (will scale pixels between 0 and 1 and then will normalize each channel with respect to the ImageNet dataset)
            img = cv2.resize(img, (width, height))
            img = img.astype(np.float32)
            img = img / 255.0
            
        # cv2.imshow(imgNorm+'2', img)

        return img
    except Exception as e:
        print(path, e)
        im = np.zeros((height, width, 3))
        return im


def getSegmentationArr(path, nClasses, width, height):
    seg_labels = np.zeros((height, width, nClasses))
    try:
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (width, height))
        img = img[:, :, 0]
        # cv2.imshow('seg', img)

        for c in range(nClasses):
            seg_labels[:, :, c] = (img == c).astype(int)

    except Exception as e:
        print(e)

    seg_labels = np.reshape(seg_labels, (width * height, nClasses))
    return seg_labels


def imageSegmentationGenerator(images_path, segs_path, batch_size, n_classes, input_height, input_width, output_height,
                               output_width):

    images = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
    segmentations = [f for f in os.listdir(segs_path) if os.path.isfile(os.path.join(segs_path, f))]
    images.sort()
    segmentations.sort()
    for i in range(len(images)):
        images[i] = images_path + images[i]
        segmentations[i] = segs_path + segmentations[i]

        # img = cv2.imread(images[i], cv2.IMREAD_COLOR)
        # cv2.imshow('image-origin',img)
        # img = cv2.imread(segmentations[i], cv2.IMREAD_COLOR)
        # cv2.imshow('image-seg',img)

    # Make sure the length of 'images' and 'segmentations' is the same.
    assert len(images) == len(segmentations)
    for im, seg in zip(images, segmentations):
        # Make sure the image name is the same.
        assert (im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])

    # 'itertools.cycle' EX: [a,b,c] >> [a,b,c,a,b,c,a, ... , c, a, ...]. infinite loop.
    zipped = itertools.cycle(zip(images, segmentations))

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = next(zipped)
            X.append(getImageArr(im, input_width, input_height, imgNorm='divide'))
            Y.append(getSegmentationArr(seg, n_classes, output_width, output_height))

        yield np.array(X), np.array(Y)
