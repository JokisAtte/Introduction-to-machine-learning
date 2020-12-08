import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,Conv2D,MaxPooling2D
from tensorflow.keras import layers
import keras
import numpy as np
import pickle
from keras.datasets import cifar10
from skimage.transform import rescale, resize, downscale_local_mean

def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape[1:])
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    shape = x_train.shape[1:]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential(
        [
            Flatten(input_shape=shape),
            Dense(20, activation='relu'),
            Dense(10, activation='sigmoid')
        ]
    )

    keras.optimizers.SGD(lr=0.7)
    model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, verbose=2)
    print("")
    print(model.evaluate(x_test,y_test, verbose=2))
    print("Bayesian classifier was 15.81% accurate and 1-NN classifier was 21% accurate")
    print("Maybe creating better model would yield better results.")
main()