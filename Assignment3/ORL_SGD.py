# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:12:06 2019

@author: anirbanhp
"""

from __future__ import print_function

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD
from matplotlib import pyplot as plt
from keras import regularizers

batch_size = 128
num_classes = 20
epochs = 25

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

data = np.load('ORL_faces.npz')
x_train = data['trainX']
y_train = data['trainY']
x_test = data['testX']
y_test = data['testY']

image_size = 10304 
x_train = x_train.reshape(x_train.shape[0], image_size) 
x_test = x_test.reshape(x_test.shape[0], image_size)
x_train = x_train / 255
x_test = x_test / 255
x_train.astype(float)
x_test.astype(float)
y_train.astype(float)
y_test.astype(float)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# MLP I - SGD
model1 = Sequential()
model1.add(Dense(512, activation='relu', input_shape=(image_size,)))
model1.add(Dropout(0.2))
model1.add(Dense(2048, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(num_classes, activation='softmax'))

model1.summary()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)

model1.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model1.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model1.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(history.history['loss'], 'b--', label='MLP I - SGD')
#plt.plot(history.history['val_loss'],'b', label='MLP I - SGD' )
# MLP II - NESTEROV
model2 = Sequential()
model2.add(Dense(512, activation='relu', input_shape=(image_size,)))
model2.add(Dropout(0.2))
model2.add(Dense(2048, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(num_classes, activation='softmax'))

model2.summary()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model2.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model2.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model2.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(history.history['loss'], 'r--', label='MLP II - NESTEROV')
#plt.plot(history.history['val_loss'], 'r', label='MLP II - NESTEROV')

#MLP III L1 NESTEROV

model3 = Sequential()
model3.add(Dense(512, activation='relu', input_shape=(image_size,)))
model3.add(Dropout(0.2))
model3.add(Dense(2048, activation='relu',kernel_regularizer=regularizers.l1(0.01)))
model3.add(Dropout(0.2))
model3.add(Dense(num_classes, activation='softmax'))

model3.summary()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model3.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model3.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model3.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(history.history['loss'], 'y--', label='MLP III L1 NESTEROV')
#plt.plot(history.history['val_loss'],'y', label='MLP III L1 NESTEROV')
# MLP IV L2 NESTEROV
model4 = Sequential()
model4.add(Dense(512, activation='relu', input_shape=(image_size,)))
model4.add(Dropout(0.2))
model4.add(Dense(2048, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model4.add(Dropout(0.2))
model4.add(Dense(num_classes, activation='softmax'))

model4.summary()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model4.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model4.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model4.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(history.history['loss'], 'g--', label='MLP IV L2 NESTEROV')
#plt.plot(history.history['val_loss'], 'g', label='MLP IV L2 NESTEROV')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("ORL faces with 2048 neurons.pdf")
plt.show()