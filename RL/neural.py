#!/usr/bin/env pyth
"""
Red neuronal
.
"""
#Arquitecctura de la red

import numpy as np
import os

import matplotlib.pyplot as plt      # MATLAB like plotting routines
import random                        # for generating random numbers

from keras.models import Sequential  # Model type to be used

import tensorflow as tf
from keras.layers.core import Dense, Dropout, Activation # Types of layers to be used in our model
from keras.utils import np_utils                         # NumPy related tools	
import reader as rd
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image


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

X_train = X_train.astype('float32')         # change integers to 32-bit floating point numbers
X_test = X_test.astype('float32')

X_train = X_train/255                              # normalize each value for each pixel for the entire vector for each input
X_test = X_test/255

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

nb_classes = 3 # numero de teclas

Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

print("Testing matrix shape", Y_train.shape)
print("Testing matrix shape", Y_test.shape)

#SANDWICH NEURONAL
model = Sequential()                                 # Linear stacking of layers

# Convolution Layer 1
model.add(Conv2D(32, (3, 3), input_shape=(120,160,3))) # 32 different 3x3 kernels -- so 32 feature maps
model.add(BatchNormalization(axis=-1))                 # normalize each feature map before activation
convLayer01 = Activation('relu')                       # activation
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
model.add(Dense(1000))                             # 587808 FCN nodes (ojo con esto)
model.add(BatchNormalization())                      # normalization
model.add(Activation('relu'))                        # activation

# Fully Connected Layer 6                       
model.add(Dropout(0.2))                              # 20% dropout of randomly selected nodes
model.add(Dense(3))                                 # final 3 FCN nodes (3 movimientos)
model.add(Activation('softmax'))                     # softmax activation

#model.summary()         #CUIDADO

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)

test_gen = ImageDataGenerator()

train_generator = gen.flow(X_train, Y_train, batch_size=25)
test_generator = test_gen.flow(X_test, Y_test, batch_size=25)

# arreglar 60000 y 128, dependiendo de la cantidad que queramos
model.fit_generator(train_generator, steps_per_epoch= len(X_train)//25, epochs=6, verbose=1, 
                    validation_data=test_generator, validation_steps= len(X_test)//25)

score = model.evaluate(X_test, Y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

velocidades = {"0":[1.0,0.0],
               "1":[0.3,1.0],
               "2":[0.3,-1.0],
               "3":[0.0,-1.0],
               "4":[0.0,1.0],
               "5":[-1.0,0.0],
               "6":[0.0,0.0],
               }

from PIL import Image

#test_image = image.load_img(path + '/frames/img190.jpg',target_size=(120,160))
#test_image = image.img_to_array(test_image)
test_image = X_test[25]
test_image = np.expand_dims(test_image,axis=0)
prediccion = model.predict(test_image)
maxima_probabilidad = max(prediccion[0])
for i in range(len(prediccion[0])):
  if maxima_probabilidad == prediccion[0][i]:
    print(velocidades["{}".format(i)])
    break
print(prediccion[0])


from keras import backend as K

def visualize(layer):
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])

    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img)
    convolutions = np.squeeze(convolutions)

    print ('Shape of conv:', convolutions.shape)

    m = convolutions.shape[2]
    n = int(np.ceil(np.sqrt(m)))

    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(15,12))
    for i in range(m):
        ax = fig.add_subplot(n,n,i+1)
        ax.imshow(convolutions[:,:,i], cmap='gray')

plt.figure()
plt.imshow(X_test[25].reshape(120,160,3), cmap='gray', interpolation='none')


#visualize(convLayer01) # visualize first set of feature maps

#visualize(convLayer02) # visualize second set of feature maps

#visualize(convLayer03)# visualize third set of feature maps

#visualize(convLayer04)# visualize fourth set of feature maps

