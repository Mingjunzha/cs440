# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np
import statistics
def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # return the trained weight and bias parameters
    label = [0]*len(train_set)
    W = [0]*len(train_set[0])
    b  = 0

    for i in range(len(train_labels)):
        if not train_labels[i]:
            label[i] = -1
        else:
            label[i] = 1

    for j in range(max_iter):
        for i in range(len(train_set)):
            image = train_set[i]
            r = np.sign(np.dot(W, image)+ b)

            if r != label[i]:
                b += learning_rate * label[i]
                W += image * (learning_rate * label[i])

    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # Train perceptron model and return predicted labels of development set
    
    w,b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    predicted = [0]*len(dev_set)
    for i in range(len(dev_set)):
        image = dev_set[i]
        r = np.sign(np.dot(w, image) + b)
        if r == 1:
            predicted[i] = 1

    return predicted

def classifyKNN(train_set, train_labels, dev_set, k):
    predicted = [0]*len(dev_set)
    for i in range(len(dev_set)):
        nei = [(0,0)] * len(train_set)
        for j in range(len(train_set)):
            score = 0
            if train_labels[j]:
                score = 1
            nei[j] = (np.linalg.norm(dev_set[i]-train_set[j]),score)
        nei = sorted(nei, key=lambda x: x[0])[:k]
        result = [x[1] for x in nei]
        countTrue = 0
        countFalse = 0
        for c in result:
            if c == 1:
                countTrue +=1
        countFalse = k-countTrue
        if countTrue == countFalse:
            predicted[i] = 0
        else:
            predicted[i] = statistics.mode(result)
    return predicted
