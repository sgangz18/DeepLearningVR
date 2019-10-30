# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""
 
import numpy as np 

def compute_euclidean_distances( X, Y ) :
    """
    Compute the Euclidean distance between two matricess X and Y  
    Input:
    X: N-by-D numpy array 
    Y: M-by-D numpy array 
    
    Should return dist: M-by-N numpy array   
    """  
    """
    for further questions --> https://stackoverflow.com/questions/27948363/numpy-broadcast-to-perform-euclidean-distance-vectorized
    """
    #dist = np.linalg.norm(X-Y)
    threeSums = np.sum(np.square(X)[:,np.newaxis,:], axis=2) - 2 * X.dot(Y.T) + np.sum(np.square(Y), axis=1)
    dist = np.sqrt(threeSums)
    return dist
 

def predict_labels( dists, labels, k=1):
    """
    Given a Euclidean distance matrix and associated training labels predict a label for each test point.
    Input:
    dists: M-by-N numpy array 
    labels: is a N dimensional numpy array
    
    Should return  pred_labels: M dimensional numpy array
    """
    dists = np.transpose(dists)
    p_labels = np.zeros(len(dists))
    for i,dist in enumerate(dists):
        nearest_neighbor = labels[dist.argsort()[:k]]
        predictions, count = np.unique(nearest_neighbor, return_counts=True)
        p_labels[i] = predictions[np.argmax(count)]

    return p_labels
    
    
     