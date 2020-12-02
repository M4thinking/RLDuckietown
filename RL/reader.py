
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

path = 'C:/Users/Max/Desktop/RLDuckietown/RL'
X = []
Y = np.loadtxt(os.path.join(path,'vel.txt'), delimiter = ',', max_rows = 100)

for i in range(100):
    img = cv2.imread(os.path.join(path,"frames", "img{}.jpg".format(i)))
    #print('Original Dimensions : ',img.shape)
    scale_percent = 25 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    X.append(resized)
    #print('Resized Dimensions : ',resized.shape)



X = np.array(X)


print(X.shape, Y.shape)

