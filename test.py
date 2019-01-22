import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

# Generate dummy data
x_train = np.random.random((10, 100, 100, 3))
# 100张图片，每张100*100*3
y_train = keras.utils.to_categorical(np.random.randint(10, size=(10, 1)), num_classes=10)
# 100*10
x_test = np.random.random((10, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(10, 1)), num_classes=10)
# 20*100

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

import os
from datetime import datetime

checkpoint_path = "training_callback" + "/" + datetime.now().strftime('%Y%m%d%H%M%S')
try:
    os.makedirs(checkpoint_path)
except:
    print('Make directory', checkpoint_path, 'happend error.')

# checkpoint_file = "weights-best.hdf5"
checkpoint_file = checkpoint_path + "/weights-epoch-{epoch:04d}-acc-{acc:.2f}.hdf5"

if os.path.isfile(checkpoint_file):
    model.load_weights(checkpoint_file)

# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
checkpoint = ModelCheckpoint(checkpoint_file, monitor='acc', verbose=0, save_best_only=False)
# model.fit(x_train, y_train, batch_size=10, epochs=5, verbose=1, callbacks=[checkpoint])
history = model.fit(x_train, y_train, batch_size=10, epochs=5, verbose=1, callbacks=[checkpoint])
# history = model.fit(x_train, y_train, batch_size=10, epochs=5, verbose=1, callbacks=[checkpoint], validation_split=0.1, shuffle=True)
score = model.evaluate(x_test, y_test, batch_size=10)

from keras.utils import plot_model
plot_model(model, to_file= checkpoint_path + '/model.png')

print(history.history.keys())

import matplotlib.pyplot as plt
fig = plt.figure()

plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.title('Model Performance')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'], loc='lower left')
fig.savefig(checkpoint_path + '/performance.png')

# plt.plot(history.history['acc'])
# # plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# fig.savefig('accuracy.png')

# plt.plot(history.history['loss'])
# # plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='lower left')
# fig.savefig('loss.png')