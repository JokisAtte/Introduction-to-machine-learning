import pickle
import numpy as np
import math
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from random import random
from skimage.transform import rescale, resize, downscale_local_mean

def class_acc(pred,gt):
    hits = 0
    for i in range(len(pred)):
        if pred[i-1] == gt[i-1]:
            hits += 1
    return hits/len(pred)

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def cifar10_color(X):
    Xf = []
    for i in range(X.shape[0]):
        # Convert images to mean values of each color channel
        img = X[i]
        #img_8x8 = resize(img, (8, 8))
        img_1x1 = resize(img, (1, 1))
        Xf.append(img_1x1[0][0])
    return Xf

def cifar10_nxm_color(X,n,m):
    Xf = []
    for i in range(X.shape[0]):
        img=X[i]
        img_2x2 = resize(img, (n, m))
        Xf.append(img_2x2[0][0])
    return Xf

def cifar_10_naivebayes_learn(Xf,Y):
    X_mean = np.zeros((10000, 3))
    label_r = {}
    label_g = {}
    label_b = {}
    for i in (range(10)):
        label_r[i] = []
        label_g[i] = []
        label_b[i] = []
    for i in range(len(Xf)):
        img_1x1 = Xf[i]
        r_vals = img_1x1[0].reshape(1 * 1)
        g_vals = img_1x1[1].reshape(1 * 1)
        b_vals = img_1x1[2].reshape(1 * 1)
        #Store all r g and b values for later sigma calculations
        label_r[Y[i]].append(r_vals[0])
        label_g[Y[i]].append(g_vals[0])
        label_b[Y[i]].append(b_vals[0])
        mu_r = r_vals.mean()
        mu_g = g_vals.mean()
        mu_b = b_vals.mean()
        X_mean[i, :] = (mu_r, mu_g, mu_b)
    x_mu_means = {}
    for i in range(10):
        x_mu_means[i] = []
    for i in range(len(X_mean)):
        label = Y[i]
        x_mu_means[label] = X_mean[i]
    sigmas = {}
    for i in range(10):
        sigmas[i] = []
    for i in range(len(Y)):
        #Calculate sigmas for each channel for each label
        sigma_r = np.std(label_r[Y[i]])
        sigma_g = np.std(label_g[Y[i]])
        sigma_b = np.std(label_b[Y[i]])
        sigmas[Y[i]] = [sigma_r, sigma_g, sigma_b]
    return x_mu_means, sigmas

def cifar10_classifier_naivebayes(x, mu, sigma, p = 0.1):
    #P=0.1 works with cifar dataset, because there is always 10% of each label
    y_pred = []
    for i in range(len(x)):
        all_results = []
        for z in range(len(mu)):
            temp = normpdf(x[i][0], mu[z][0], sigma[z][0]) *\
                   normpdf(x[i][1], mu[z][1], sigma[z][1]) *\
                   normpdf(x[i][2], mu[z][2], sigma[z][2]) * p
            all_results.append(temp)
        y_pred.append(all_results.index(max(all_results)))
    return y_pred

def cifar10_classifier_bayes(x,mu,sigma, p = 0.1):
    y_pred = []
    for i in range(len(x)):
        all_results = []
        for z in range(len(mu)):
            temp = multivariate_normal.pdf(x[i], mean = mu[z][0], cov = sigma[z])
            all_results.append(temp)
        y_pred.append(all_results.index(max(all_results)))
    return y_pred

def cifar10_bayes_learn(Xf, Y):
    X_mean = np.zeros((10000, 3))
    for i in range(len(Xf)):
        img_1x1 = Xf[i]
        r_vals = img_1x1[0]
        g_vals = img_1x1[1]
        b_vals = img_1x1[2]
        mu_r = r_vals.mean()
        mu_g = g_vals.mean()
        mu_b = b_vals.mean()
        X_mean[i, :] = (mu_r, mu_g, mu_b)
    x_mu_means = {}
    for i in range(10):
        x_mu_means[i] = []
    for i in range(len(X_mean)):
        label = Y[i]
        x_mu_means[label].append(X_mean[i])
    x_mu_vectors = {}
    for i in range(10):
        x_mu_vectors[i] = []
    for i in range(len(x_mu_means)):
        arr = np.empty((len(x_mu_means[i]),3))
        for z in range(len(x_mu_means[i])):
            arr[z, :] = x_mu_means[i][z]
        x_mu_vectors[i] = arr
    covs = {}
    for i in range(10):
        covs[i] = []
    for i in range(len(x_mu_vectors)):
        vec_r = np.reshape(x_mu_vectors[i][:, 0],(len(x_mu_vectors[i]),))
        vec_g = np.reshape(x_mu_vectors[i][:, 1],(len(x_mu_vectors[i]),))
        vec_b = np.reshape(x_mu_vectors[i][:, 2],(len(x_mu_vectors[i]),))
        covs[i] = np.cov([vec_r,vec_b,vec_g])
    return x_mu_means, covs

def normpdf(x, mean, sd):
    denom = (2*math.pi*sd)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*sd))
    return num/denom

#datadict = unpickle('../cifar-10-batches-py/data_batch_1')
datadict = unpickle('cifar-10-batches-py/test_batch')

X = datadict["data"]
Y = datadict["labels"]
labeldict = unpickle('cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
Y = np.array(Y)


##UNCOMMENT THIS FOR TASKS 1 and 2##

#Xf = cifar10_color(X)
#mu_means, sigmas = cifar_10_naivebayes_learn(Xf, Y)
#y_pred = cifar10_classifier_naivebayes(Xf, mu_means, sigmas)
#print("Accuracy for task 1 classifier: ", class_acc(y_pred, Y))
#mu_means, sigmas = cifar10_bayes_learn(Xf, Y)
#y_pred = cifar10_classifier_bayes(Xf, mu_means, sigmas)
#print("Accuracy for task 2 classifier: ", class_acc(y_pred, Y))
#print("Task 2 classifier is better. It uses more data to train")


#Task 3
sizes = [2,4,8,16,32]
results = []
for i in sizes:
    Xf2 = cifar10_nxm_color(X, i, i)
    mu_means, sigmas = cifar10_bayes_learn(Xf2, Y)
    y_pred = cifar10_classifier_bayes(Xf2, mu_means, sigmas)
    results.append(class_acc(y_pred, Y))
plt.plot(sizes,results)
plt.show()
