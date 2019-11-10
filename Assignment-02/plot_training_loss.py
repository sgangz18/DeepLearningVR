# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 22:50:22 2019

@author: anirbanhp
"""

from load_mnist import * 
import matplotlib.pyplot as plt
import numpy
from sklearn.neural_network import MLPClassifier
#from keras.layers import Dense
#from keras.models import Sequential


X_train, y_train = load_mnist('training'  )
X_test, y_test = load_mnist('testing'   )

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
'''
model = Sequential()
model.add(Dense(12, input_dim = 8, activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sgd'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

data = model.fit(X_train, y_train, validation_split=0.33, epochs= 150, batch_size=10, verbose=0)

print(data.data.keys())
'''
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

data = mlp.fit(X_train, y_train)

plt.plot(data['loss'])
plt.plot(data['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('iterations')
plt.legend(['train', 'test'], loc='upper left')
plt.show()