
#Las necesesarias para ejecutar el programa
import sys
import argparse
import numpy as np
import cv2
import os

#La red/Rl            
import matplotlib.pyplot as plt      # MATLAB like plotting routines
import matplotlib.image as img
import random                        # for generating random numbers

path = '/Users/tamarahan/RLDuckietown/RL'
X = []
Y_ = np.loadtxt(os.path.join(path,'vel.txt'), delimiter = ',', max_rows = 200)
Y = []
for i in range(200):
    img = cv2.imread(os.path.join(path,"frames", "img{}.jpg".format(i)))
    #print('Original Dimensions : ',img.shape)
    scale_percent = 25 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    X.append(resized)
    comp = (Y_[i][0],Y_[i][1])
    if comp == (0.0,-1.0):
        Y_[i] = [0]
    elif comp == (0.0,1.0):
        Y_[i] = [1]
    elif comp == (0.3,-1.0):
        Y_[i] = [2]
    elif comp == (0.3,1.0):
        Y_[i] = [3]
    elif comp == (1.0,0.0):
        Y_[i] = [4]
    elif comp == (-1.0,0.0):
        Y_[i] = [5]
    elif comp == (0.0,0.0):
        Y_[i] = [6]
        
    Y.append(Y_[i][0])
    #print('Resized Dimensions : ',resized.shape)

X = np.array(X)
Y = np.array(Y)

velocidades = {"0":[0.0,-1.0],
               "1":[0.0,1.0],
               "2":[0.3,-1.0],
               "3":[0.3,1.0],
               "4":[1.0,0.0],
               "5":[-1.0,0.0],
               "6":[0.0,0.0],
               }

#velocidades['0'] = [0.0,-1.0]


#print(X.shape, Y.shape)

