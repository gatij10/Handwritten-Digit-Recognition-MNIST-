import numpy as np
from matplotlib import pyplot as plt

import keras
from keras.datasets import mnist

(X_train,y_train),(X_test,y_test)=mnist.load_data()

print('Training data {}:'.format(X_train.shape))
print('Testing data {}:'.format(X_test.shape))

from keras.utils import np_utils

num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')


X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam

def first_model():


    model=Sequential()

    model.add(Dense(50,activation='relu',input_shape=(784,)))
    model.add(Dense(50,activation='relu'))


    model.add(Dense(num_classes,activation='softmax'))

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model


model=first_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=100, verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
