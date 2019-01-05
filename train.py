import Models
import LoadBatches
import os
import tensorflow as tf

train_images_path = 'data/train/'
train_segs_path = 'data/label/'
train_batch_size = 1
n_classes = 2

input_height = 384
input_width = 384
output_height = 384
output_width = 384

validate = False
save_weights_path = 'results/'
# epochs = 100
EPOCHS = 100
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

model = Models.Unet(n_classes, input_height=input_height, input_width=input_width, nChannels=3)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer_name,
              metrics=['accuracy'])

print("Model output shape", model.output_shape)
model.summary()

# === 保存和恢復模型 ===
checkpoint_path = "training_callback/cp-{epoch:04d}-{val_acc:.2f}.ckpt"
# checkpoint_path = "training_callback/cp.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# if os.path.isfile(checkpoint_path):
#     model.load_weights(checkpoint_path)
#     print("Load last time weigths file.")
# ==========================

TRAIN_IMG_NUM = len([name for name in os.listdir(train_images_path) if os.path.isfile(os.path.join(train_images_path, name))])
TRAIN_SEG_NUM = len([name for name in os.listdir(train_segs_path) if os.path.isfile(os.path.join(train_segs_path, name))])
if TRAIN_IMG_NUM != TRAIN_SEG_NUM:
    raise Exception('TRAIN_IMG_NUM = ', TRAIN_IMG_NUM , ' ; TRAIN_SEG_NUM = ', TRAIN_SEG_NUM, '. Is not equal.')

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
    history = model.fit_generator(G, TRAIN_IMG_NUM//train_batch_size, epochs=EPOCHS, callbacks=[cp_callback])
    # model.fit_generator(G, 1//train_batch_size, epochs=EPOCHS)
    model.save_weights(save_weights_path + "model_epochs." + str(EPOCHS))
    model.save(save_weights_path + "model_epochs" + str(EPOCHS) + ".h5")
else:
    model.fit_generator(G, steps_per_epoch=STEPS_PER_EPOCH, validation_data=G2, validation_steps=200)
    model.save_weights(save_weights_path + "model_steps." + str(STEPS_PER_EPOCH))
    model.save(save_weights_path + "model_steps" + str(STEPS_PER_EPOCH) + ".h5")

from keras.utils import plot_model
plot_model(model, to_file='model.png')

print(history.history.keys())

fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower left')
#
fig.savefig('performance.png')