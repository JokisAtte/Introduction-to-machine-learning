import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random, randint
from math import sqrt

def class_acc(pred,gt):
    hits = 0
    for i in range(len(pred)):
        if pred[i-1] == gt[i-1]:
            hits += 1
    return hits/len(pred)


def cifar10_classifier_random(x):
    index = randint(0, 9)
    return index


def cifar10_classifier_1nn(x, trdata, trlabels):
    distances = {}
    labeled = []
    for n in range(len(x)):
        distances[n] = []
        for i in range(len(trdata)):
            distances[n].append([euclidean_distance(x[n], trdata[i]), trlabels[i]])
        distances[n].sort()
        label = distances[n][0][1]
        labeled.append(label)
    return labeled

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (int(row1[i]) - int(row2[i]))**2
    return sqrt(distance)


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

datadict = unpickle('../cifar-10-batches-py/data_batch_1')

X = datadict["data"]
Y = datadict["labels"]

labeldict = unpickle('../cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y = np.array(Y)

pred_labels = []
for i in range(len(datadict["labels"])):
    pred_labels.append(cifar10_classifier_random(Y))
print("Random classes:")
print(class_acc(pred_labels,Y))
datadict_dev = datadict["data"][:100]
train_data = datadict["data"][101:201]
trainlabels = Y[101:201]
nn = cifar10_classifier_1nn(datadict_dev,train_data,trainlabels)
print(" ")
print("1-NN:")
print(class_acc(nn,Y[:100]))

#for i in range(X.shape[0]):
#    # Show some images randomly
#    if random() > 0.999:
#        plt.figure(1)
#        plt.clf()
##       plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
#        plt.pause(1)