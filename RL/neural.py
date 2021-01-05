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


#Se descarga conjunto de entrenamiento

X_train = rd.X  
Y_train = rd.Y  

X_train = X_train.astype('float32')                # cambiar enteros a float de 32 bits
X_train = X_train/255                              # normalizar cada valor de cada píxel para todo el vector de cada entrada

nb_classes = 3 # numero de teclas Q-W-E

Y_train = np_utils.to_categorical(Y_train, nb_classes)

print("Training X matrix shape", X_train.shape)
print("Training Y matrix shape", Y_train.shape)

#Modelo de capas de redes neuronales
model = Sequential()                                   # Apilado lineal de capas

# Capa de convolución 1
model.add(Conv2D(32, (3, 3), input_shape=(120,160,3))) # 32 kernels de 3x3 diferentes --> 32 feature map
model.add(BatchNormalization(axis=-1))                 # normalizar cada feature map antes de la activación
convLayer01 = Activation('relu')                       # activación
model.add(convLayer01)

# Capa de convolución 2
model.add(Conv2D(32, (3, 3)))                        # 32 kernels de 3x3 diferentes --> 32 mapas nuevos
model.add(BatchNormalization(axis=-1))               # normalizar cada feature map antes de la activación
model.add(Activation('relu'))                        # activacion
convLayer02 = MaxPooling2D(pool_size=(2,2))          # agrupa los valores máximos en un kernel 2x2
model.add(convLayer02)

# Capa de convolución 3
model.add(Conv2D(64,(3, 3)))                         # 64 kernels de 3x3 diferentes --> 64 mapas nuevos
model.add(BatchNormalization(axis=-1))               # normalizar cada feature map antes de la activación
convLayer03 = Activation('relu')                     # activación
model.add(convLayer03)

# Capa de convolución 4
model.add(Conv2D(64, (3, 3)))                        # 64 kernels de 3x3 diferentes --> 64 mapas nuevos
model.add(BatchNormalization(axis=-1))               # normalizar cada feature map antes de la activación
model.add(Activation('relu'))                        # Funcion de activacion
convLayer04 = MaxPooling2D(pool_size=(2,2))          # agrupa los valores máximos en un kernel 2x2
model.add(convLayer04)
model.add(Flatten())                                 # aplanado final : 4x4x64 matriz de salida --> vector de 1024-largo

# Capa completamente conectada 5
model.add(Dense(1000))                               # 1000 FCN nodes (ojo con esto)
model.add(BatchNormalization())                      # normalización
model.add(Activation('relu'))                        # activación

# Capa completamente conectada 6                       
model.add(Dropout(0.2))                              # 20% déficit de nodos seleccionados al azar
model.add(Dense(3))                                  # 3 movimientos posibles finales
model.add(Activation('softmax'))                     # activacion softmax 

#model.summary()                                     #CUIDADO, para ver el resumen del modelo

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Se entrega dirección donde se guarda el modelo (modificar)
path = '/Users/HP/Desktop/RLDuckietown/RL'
# Se corre el modelo con el conjunto de entrenamiento
#En este caso en particular no se uso un conjunto de validación exterior
model.fit(X_train,Y_train, epochs=7 , verbose=1, batch_size=20, validation_split=0.20, validation_data=None)
model.save(os.path.join(path,"models", "nombre_modelo.h5"))
print('Modelo listo con nombre: ' + 'nombre_modelo' )

#Si se tiene un conjunto de entrenamiento exterior, es posible comparar el desempeño del modelo

#X_test = X_exterior
#Y_test = Y_exterior

#n_datos_test = len(X_test)

#print("X_test shape", X_test.shape)
#print("Y_test shape", Y_test.shape)

#X_test = X_test.reshape(n_datos_test, 640, 480, 3)

#X_test = np.array(X_test).astype('float32')

#X_test = X_test/255

#Y_test = np_utils.to_categorical(Y_test, nb_classes)

#print("Testing matrix shape", X_test.shape)
#print("Testing Y matrix shape", Y_test.shape)

#score = model.evaluate(X_test, Y_test)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])

velocidades = {"0":[1.0,0.0],  #Adelante W
               "1":[0.3,1.0],  #Derecha/Adelante E
               "2":[0.3,-1.0], #Izquierda/Adelante Q
               "3":[0.0,-1.0],
               "4":[0.0,1.0],
               "5":[-1.0,0.0],
               "6":[0.0,0.0],
               }