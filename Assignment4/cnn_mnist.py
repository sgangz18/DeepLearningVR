# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:58:58 2019

@author: Anirban

https://medium.com/octavian-ai/which-optimizer-and-learning-rate-should-i-use-for-deep-learning-5acb418f9b2
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from matplotlib import pyplot as plt
from keras import regularizers
from keras.optimizers import RMSprop, SGD
batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2),
                 activation='relu',strides=2,
                 input_shape=input_shape))
model.add(Conv2D(64, (2, 2),strides=2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95)
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=adadelta,
              metrics=['accuracy'])

history=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(history.history['loss'], 'y--', label='CNN ')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.savefig("CNN_sgd_0.1.pdf")

#add regularization to cnn


model2 = Sequential()
model2.add(Conv2D(32, kernel_size=(2, 2),
                 activation='relu',strides=2,
                 input_shape=input_shape))
model2.add(Conv2D(64, (2, 2),strides=2, activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.5))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model2.add(Dense(num_classes, activation='softmax',\
                 kernel_regularizer=regularizers.l2(0.01),\
                 bias_regularizer=regularizers.l2(0.01)))

model2.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=adadelta,
              metrics=['accuracy'])

history2=model2.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score2 = model2.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score2[0])
print('Test accuracy:', score2[1])
plt.plot(history2.history['loss'], 'r--', label='CNN_weightdecay')
plt.savefig("MNIST_adadelta.pdf")

plt.legend()
plt.show()