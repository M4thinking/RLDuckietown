#!/usr/bin/env pyth
"""
Este programa permite abrir el conjunto de entrenamiento y leerlo.

"""

import os
import numpy as np
import cv2

#Se entrega el path, la cantidad de datos (modificar)
n_datos = 100
#Poner dirección donde se encuentra carpeta frames y vel.txt
path = '/Users/HP/Desktop/RLDuckietown/RL'

#Por defecto debe crear carpeta frames en mismo archivo donde se encuentra reader.py
#El archivo de texto es creado automaticamente en este mismo archivo

#Se obtiene un vector X con todos los frames y otro Y con todas las velocidades
X = []
Y = []
#Arreglo auxiliar para traspasar las variables lineales y angulares a un solo valor
Y_ = np.loadtxt(os.path.join(path,'vel.txt'), delimiter = ',', max_rows = n_datos)
for i in range(n_datos):
    #Se lee la imagen i-ésima
    img = cv2.imread(os.path.join(path,"frames", "img{}.jpg".format(i)))
    #Se reescala la imagen un 25%  640 x 480 -> 160 x 120
    scale_percent = 25 # porcentaje de la imagen original
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    X.append(resized)
    #Se pasan las componentes de la velocidad a un solo valor representativo
    comp = (Y_[i][0],Y_[i][1])
    #En este caso solo se utilizó Q-W-E para realizar los movimientos
    if comp == (1.0,0.0):
        Y_[i] = [0]
    elif comp == (0.3,1.0):
        Y_[i] = [1]
    elif comp == (0.3,-1.0):
        Y_[i] = [2]
    #Se agrega esta componente representativa al vector Y
    Y.append(Y_[i][0])

print("Procesamiento terminado")
#Se cambia de lista de python a lista de NumPy
X = np.array(X)
Y = np.array(Y)

#Diccionario de velocidades
velocidades = {"0":[1.0,0.0],  #Adelante W
               "1":[0.3,1.0],  #Derecha/Adelante E
               "2":[0.3,-1.0], #Izquierda/Adelante Q
               "3":[0.0,-1.0],
               "4":[0.0,1.0],
               "5":[-1.0,0.0],
               "6":[0.0,0.0],
               }


