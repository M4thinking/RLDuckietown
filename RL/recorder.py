#!/usr/bin/env pyth
"""
Este programa permite grabar el entorno de Duckietwown (Frames y Velocidades).

"""
#Se importan librerias necesesarias para ejecutar el ambiente
import sys
import argparse
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
import numpy as np
import cv2
import os          

#Función para mover el duckiebot 
def mov_duckiebot(key):
    # La acción de Duckiebot consiste en dos valores:
    # velocidad lineal y velocidad de giro
    actions = {ord('w'): np.array([1.0, 0.0]),
               ord('s'): np.array([-1.0, 0.0]),
               ord('a'): np.array([0.0, 1.0]),
               ord('d'): np.array([0.0, -1.0]),
               ord('q'): np.array([0.3, 1.0]),
               ord('e'): np.array([0.3, -1.0])
               }

    return actions.get(key, np.array([0.0, 0.0]))

#Definimos nuestros envirioment
if __name__ == '__main__':

    # Se leen los argumentos de entrada
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="Duckietown-udem1-v1")
    parser.add_argument('--map-name', default='udem1')
    parser.add_argument('--distortion', default=False, action='store_true')
    parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
    parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
    parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
    parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
    parser.add_argument('--seed', default=1, type=int, help='seed')
    args = parser.parse_args()

    # Definición del environment
    if args.env_name and args.env_name.find('Duckietown') != -1:
        env = DuckietownEnv(
            seed = args.seed,
            map_name = args.map_name,
            draw_curve = args.draw_curve,
            draw_bbox = args.draw_bbox,
            domain_rand = args.domain_rand,
            frame_skip = args.frame_skip,
            distortion = args.distortion,
        )
    else:
        env = gym.make(args.env_name)

    # Se reinicia el environment
    env.reset()

    #Iterador para enumeración de los datos
    i = 0
    #Se genera un archivo de texto para guardar velocidades
    archivo = open("vel.txt", 'w')
    while True:

        # Captura la tecla que está siendo apretada y almacena su valor en key
        key = cv2.waitKey(30)
        # Si la tecla es Esc, se sale del loop y termina el programa
        if key == 27:
            break
        
        action = mov_duckiebot(key)
        #Se guardan las componentes de la velocidad
        comp = (action[0],action[1])
        print(comp)
        if comp == (1.0, 0.0) or comp == (0.3, 1.0) or comp == (0.3, -1.0): #Adelante, derecha/adelante o izquierda/adelante
            # Se ejecuta la acción definida anteriormente y se retorna la observación (obs), la evaluación (reward), etc
            # obs consiste en un imagen RGB de 640 x 480 x 3
            obs, reward, done, info = env.step(action)
            #Se escriben las componentes de la velocidad en el archivo
            archivo.write(str(action[0])+","+str(action[1]) +'\n')
            #Se declara la carpeta donde se guardaran las imagenes (crear en la misma ruta que se encuentra recorder.py)
            path = 'frames'
            #Se escribe la imagen en la caperta del path
            cv2.imwrite(os.path.join(path,"img{}.jpg".format(i)), cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            i+=1
        else:#No se guarda la imagen y se sigue con la conducción
            obs, reward, done, info = env.step(action)

        if done: # done significa que el Duckiebot chocó con un objeto o se salió del camino
            print('done!')
            # En ese caso se reinicia el simulador
            env.reset()
 
        # Se muestra en una ventana llamada "patos" la observación del simulador
        cv2.imshow("patos", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    archivo.close()
# Se cierra el environment y termina el programa
env.close()
