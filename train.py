import Models
import LoadBatches
import os
import keras

train_images_path = 'data/train/'
train_segs_path = 'data/label/'
train_batch_size = 1
n_classes = 2

# input_height = 384
# input_width = 384
# output_height = 384
# output_width = 384
input_height = 768
input_width = 768
output_height = 768
output_width = 768

validate = False
save_weights_path = 'results/'
# epochs = 100
EPOCHS = 1000
STEPS_PER_EPOCH = 5
optimizer_name = 'adam'

val_images_path = ''
val_segs_path = ''
val_batch_size = 2

try:
    os.makedirs(train_images_path)
except:
    print('Make directory', train_images_path, 'happend error.')

try:
    os.makedirs(train_segs_path)
except:
    print('Make directory', train_segs_path, 'happend error.')

try:
    os.makedirs(save_weights_path)
except:
    print('Make directory', save_weights_path, 'happend error.')

# === Make Models and Compile ===
model = Models.Unet(n_classes, input_height=input_height, input_width=input_width, nChannels=3)

model.compile(loss=Models.MixedLoss,
              optimizer=optimizer_name,
              metrics=['accuracy'])

print("Model output shape", model.output_shape)
model.summary()

# === 保存和恢復模型 ===
import os
from datetime import datetime

checkpoint_path = "training_callback" + "/" + datetime.now().strftime('%Y%m%d%H%M%S')
try:
    os.makedirs(checkpoint_path)
except:
    print('Make directory', checkpoint_path, 'happend error.')

checkpoint_file = checkpoint_path + "/weights-epoch-{epoch:04d}-acc-{acc:.2f}.hdf5"

# if os.path.isfile(checkpoint_file):
#     model.load_weights(checkpoint_file)

from keras.callbacks import ModelCheckpoint
cp_callback = ModelCheckpoint(checkpoint_file, monitor='acc',save_weights_only=True, verbose=1)

# === Check length of input image is same with length of seg images ===
TRAIN_IMG_NUM = len([name for name in os.listdir(train_images_path) if os.path.isfile(os.path.join(train_images_path, name))])
TRAIN_SEG_NUM = len([name for name in os.listdir(train_segs_path) if os.path.isfile(os.path.join(train_segs_path, name))])
if TRAIN_IMG_NUM != TRAIN_SEG_NUM:
    raise Exception('TRAIN_IMG_NUM = ', TRAIN_IMG_NUM , ' ; TRAIN_SEG_NUM = ', TRAIN_SEG_NUM, '. Is not equal.')

# === Make a image generator ===

G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_batch_size, n_classes,
                                           input_height, input_width, output_height, output_width)
if validate:
    G2 = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_batch_size, n_classes, input_height,
                                                input_width, output_height, output_width)
 
# if not validate:
#     for ep in range(epochs):
#         print('current epoch:', ep)
#         model.fit_generator(G, 400//train_batch_size, epochs=1)
#         model.save_weights(save_weights_path + "model_" + str(ep) + ".h5")
#         model.save(save_weights_path + "model_" + str(ep) + ".json")
# else:
#     for ep in range(epochs):
#         model.fit_generator(G, 512, validation_data=G2, validation_steps=200, epochs=1)
#         model.save_weights(save_weights_path + "." + str(ep))
#         model.save(save_weights_path + "model_" + str(ep) + ".h5")

if not validate:
    # === Steps_Per_Epoch
    # model.fit_generator(G, steps_per_epoch=STEPS_PER_EPOCH)
    # model.save_weights(save_weights_path + "model_steps." + str(STEPS_PER_EPOCH))
    # model.save(save_weights_path + "model_steps" + str(STEPS_PER_EPOCH) + ".h5")

    # === Epochs 
    # history = model.fit_generator(G, TRAIN_IMG_NUM//train_batch_size, epochs=EPOCHS, callbacks=[cp_callback])
    model.fit_generator(G, TRAIN_IMG_NUM//train_batch_size, epochs=EPOCHS, callbacks=[cp_callback])
    # model.fit_generator(G, 1//train_batch_size, epochs=EPOCHS)
    model.save_weights(save_weights_path + "model_epochs." + str(EPOCHS))
    model.save(save_weights_path + "model_epochs" + str(EPOCHS) + ".h5")
else:
    model.fit_generator(G, steps_per_epoch=STEPS_PER_EPOCH, validation_data=G2, validation_steps=200)
    model.save_weights(save_weights_path + "model_steps." + str(STEPS_PER_EPOCH))
    model.save(save_weights_path + "model_steps" + str(STEPS_PER_EPOCH) + ".h5")
