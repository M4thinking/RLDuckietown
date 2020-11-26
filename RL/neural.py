#!/usr/bin/env pyth
"""
Este programa permite mover
.
"""
#Arquitecctura de la red

import numpy as np
import os

import matplotlib.pyplot as plt      # MATLAB like plotting routines
import random                        # for generating random numbers

from keras.models import Sequential  # Model type to be used

from keras.layers.core import Dense, Dropout, Activation # Types of layers to be used in our model
from keras.utils import np_utils                         # NumPy related tools	
import reader as rd
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Flatten
from keras.layers.normalization import BatchNormalization

largo = len(rd.Y) 
corto = int(largo * 0.75)


X_train = rd.X[:corto]
X_test = rd.X[corto:]
Y_train = rd.Y[:corto]
Y_test = rd.Y[corto:]


#print("X_train shape", X_train.shape)
#print("y_train shape", Y_train.shape)
#print("X_test shape", X_test.shape)
#print("y_test shape", Y_test.shape)

#X_train = X_train.reshape(largo, 640, 480, 3) #add an additional dimension to represent the single-channel
#X_test = X_test.reshape(corto, 640, 480, 3)

#X_train = X_train.astype('float32')         # change integers to 32-bit floating point numbers
#X_test = X_test.astype('float32')

#X_train /= 255                              # normalize each value for each pixel for the entire vector for each input
#X_test /= 255

#print("Training matrix shape", X_train.shape)
#print("Testing matrix shape", X_test.shape)

nb_classes = 6 # numero de teclas

Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)



#SANDWICH NEURONAL
model = Sequential()                                 # Linear stacking of layers

# Convolution Layer 1
model.add(Conv2D(32, (3, 3), input_shape=(640,480,3))) # 32 different 3x3 kernels -- so 32 feature maps
model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
convLayer01 = Activation('relu')                     # activation
model.add(convLayer01)

# Convolution Layer 2
model.add(Conv2D(32, (3, 3)))                        # 32 different 3x3 kernels -- so 32 feature maps
model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
model.add(Activation('relu'))                        # activation
convLayer02 = MaxPooling2D(pool_size=(2,2))          # Pool the max values over a 2x2 kernel
model.add(convLayer02)

# Convolution Layer 3
model.add(Conv2D(64,(3, 3)))                         # 64 different 3x3 kernels -- so 64 feature maps
model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
convLayer03 = Activation('relu')                     # activation
model.add(convLayer03)

# Convolution Layer 4
model.add(Conv2D(64, (3, 3)))                        # 64 different 3x3 kernels -- so 64 feature maps
model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
model.add(Activation('relu'))                        # activation
convLayer04 = MaxPooling2D(pool_size=(2,2))          # Pool the max values over a 2x2 kernel
model.add(convLayer04)
model.add(Flatten())                                 # Flatten final 4x4x64 output matrix into a 1024-length vector

# Fully Connected Layer 5
model.add(Dense(512))                                # 512 FCN nodes
model.add(BatchNormalization())                      # normalization
model.add(Activation('relu'))                        # activation

# Fully Connected Layer 6                       
model.add(Dropout(0.2))                              # 20% dropout of randomly selected nodes
model.add(Dense(6))                                 # final 10 FCN nodes
model.add(Activation('softmax'))                     # softmax activation

#model.summary()         #CUIDADO
