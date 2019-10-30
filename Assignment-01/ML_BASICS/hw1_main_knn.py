# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:42:21 2019

@author: anirbanhp
"""

 
from load_mnist import load_mnist
import hw1_knn  as mlBasics  
import numpy as np 
'''
(a) (0.5 point) In hw1 demo knn.py you are given a subset of the dataset with two
classes 0 and 1. Following the instructions given in hw1 demo knn.py load data
for \ALL CLASSES" ( 60000 training images and 10000 test images).
'''

X_train, y_train = load_mnist('training' )
X_test, y_test = load_mnist('testing'   )

#x0 = np.arange(X_train[1])

new_X = np.empty((1000,28,28), dtype='int')
new_y = np.empty(1000)

k, l = 0, 100
for i in range (1, 10):
    temp, temp_lbl = load_mnist('training', [i])
    random = np.random.randint(temp.shape[0], size=100)
    random_X = temp[random]
    random_y = temp_lbl[random]
    #print(random_X.shape)
    #print(random_y.shape)
    #np.append(new_X, random_X)
    #for k,l in range (1, 100):
    new_X[k:l] = random_X[0:100]
    new_y[k:l] = random_y[0:100]
    k, l = l, l+100

# Reshape images
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

dists =  mlBasics.compute_euclidean_distances(X_train,X_test) 

y_test_pred = mlBasics.predict_labels(dists, y_train_sample, k=1)

y_test_pred_5 = mlBasics.predict_labels(dists, y_train_sample, k=5)