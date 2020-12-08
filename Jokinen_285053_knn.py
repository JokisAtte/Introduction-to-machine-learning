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

def classifier_knn(x, trdata, trlabels,k=1):
    distances = {}
    labeled = []
    for n in range(len(x)):
        distances[n] = []
        for i in range(len(trdata)):
            distances[n].append([euc_distance(x[n], trdata[i]), trlabels[i]])
        distances[n].sort()
        labels = []
        for i in range(k):
            labels.append(distances[n][i][1])
        label = max(set(labels), key=labels.count)
        labeled.append(label)
    return labeled

def main():
    X_test = np.loadtxt("X_test.txt")
    Y_test = np.loadtxt("Y_test.txt")
    X_train = np.loadtxt("X_train.txt")
    Y_train = np.loadtxt("Y_train.txt")

    k_values = [1, 2, 3, 5, 10, 20]
    print("K-nn classifier accuracies:")
    for i in k_values:
        y_pred = classifier_knn(X_test, X_train, Y_train, i)
        print("K-nn classifier accuracy when K =", i, ":", class_acc(y_pred,Y_test))
    return

main()