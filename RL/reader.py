
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
Y = np.loadtxt(os.path.join(path,'vel.txt'), delimiter = ',')

for i in range(len(Y)):
	X.append(cv2.imread(os.path.join(path,"frames", "img{}.jpg".format(i))))

X = np.array(X)
#print(X.shape, Y.shape)

