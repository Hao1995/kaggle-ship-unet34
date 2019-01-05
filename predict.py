import Models
import LoadBatches
import glob
import cv2
import numpy as np

def predict_segmentation():
    n_classes = 2
    images_path = 'data/test/1img/'
    input_width = 768
    input_height = 768
    output_height = 768
    output_width = 768
    # input_width = 64
    # input_height = 96
    # output_width = 64
    # output_height = 96

    EPOCHS = 5
    optimizer_name = 'adam'

    output_path = 'data/seg_results/'

    m = Models.Unet(n_classes, input_height=input_height, input_width=input_width, nChannels=3)

    m.load_weights("results/model_" + 'epochs' + str(EPOCHS) + ".h5")
    # m.load_weights("training_callback/cp-0024.ckpt")
    
    # m.load_weights("results/model_person_99.h5")
    m.compile(loss='categorical_crossentropy',
              optimizer=optimizer_name,
              metrics=['accuracy'])

    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
    images.sort()

    colors = [(0, 0, 0), (255, 255, 255)]

    for imgName in images:
        imgName = imgName.replace('\\', '/')
        outName = imgName.replace(images_path, output_path)
        X = LoadBatches.getImageArr(imgName, input_width, input_height, imgNorm='divide')
        pr = m.predict(np.array([X]))[0]
        pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
        seg_img = np.zeros((output_height, output_width, 3))
        for c in range(n_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
        seg_img = cv2.resize(seg_img, (input_width, input_height))
        cv2.imwrite(outName, seg_img)


predict_segmentation()