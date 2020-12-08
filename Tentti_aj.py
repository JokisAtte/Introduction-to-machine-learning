import numpy as np
from math import sqrt

def class_acc(pred,gt):
    hits = 0
    for i in range(len(pred)):
        if pred[i-1] == gt[i-1]:
            hits += 1
    return hits/len(pred)

def euc_distance(a,b):
    distance = 0.0
    for i in range(len(a)-1):
        distance += (int(a[i]) - int(b[i]))**2
    return sqrt(distance)

def classifier_1nn(x, trdata, trlabels):
    distances = {}
    labeled = []
    for n in range(len(x)):
        distances[n] = []
        for i in range(len(trdata)):
            distances[n].append([euc_distance(x[n], trdata[i]), trlabels[i]])
        distances[n].sort()
        label = distances[n][0][1]
        labeled.append(label)
    return labeled

def main():
    # Load data
    X_test = np.loadtxt("X_test.txt")
    Y_test = np.loadtxt("Y_test.txt")
    X_train = np.loadtxt("X_train.txt")
    Y_train = np.loadtxt("Y_train.txt")

    Y_pred = classifier_1nn(X_test, X_train, Y_train)
    print("1-nn classifier accuracy: ", class_acc(Y_pred,Y_test))
    return

main()