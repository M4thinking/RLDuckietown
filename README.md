# RLDuckietown
Reinforcement Learning para conducción autónoma-Duckietown.

**Orden de creación de archivos**

- reader.py : Este programa permite abrir el conjunto de entrenamiento y leerlo.

- recorder.py : Este programa permite grabar el entorno de Duckietwown (Frames y Velocidades).

- neural.py : Red neuronal convolucional para entrenar el modelo

- autoduck.py : Este programa permite mover autónomamente el Duckiebot.

**Recomendaciones:**

Dentro del README.md de la caperta models, se encuentra un link al modelo de prueba generado durante la confección de este proyecto. Una vez descargado el modelo de prueba, dejar en carpeta models.

Cambiar la dirección de los paths de reader.py, neural.py y autoduck.py a la dirección del ordenador del usuario.

Descargar las librerias de TensorFlow y Keras.

Ejecutar recorder.py bajo el entorno de ejecución de gym-duckietown. Ahora podrá manejar libremente por el mapa por defecto que usted prefiera y tomar captura de los frames y las componentes de velocidad usadas. Durante el proyecto se determinó que solo se guardarán las velocidades asociadas a las teclas Q-W-E, por tanto no debiese tener necesidad de utilizar otras teclas para moverse por el mapa. 

Una vez hecho esto por el tiempo que se estime conveniente, y en su defecto, para que el entrenamiento tenga sentido, probar con intervalos de 3000 a 20000 fotos dependiendo de lo que el ordenador pueda soportar más adelante en el entrenamiento. Una vez cerrado el vizualizador del entorno, se verán en la carpeta RL los frames guardados en la carpeta frames y las componentes de las velocidades en un archivo de texto vel.txt. Esta es toda la información necesaria para correr el siguiente programa.

A continuación, dirijirse a reader.py , modificar el path (si no se ha hecho aun) e indicar el número de datos que se quieren procesar en el entrenamiento.

Ahora, dirijirse a neural.py donde se encontrará la estructura de una red neuronal convolucional, la cual fue utilizada para procesar las imagenes asociadas a sus respectivas velocidades. Una capa de convolucion es basicamente un preprocesamiento de los frames, haciéndolos pasar por filtros representados por matrices. Así, se aplica una operación matricial a cada mapa de características, generando nuevas imagenes alteradas que servirán para detectar patrones. Aquí se hacen tambien operaciones como Max-Pooling, que genera una disminución en las dimensiones de la imagen, quedandose con los pixeles más representativos de la imagen en cuestión. Se aplican funciones de activación, como 'relu', que elimina todos los valores negativos dentro los frames, o 'softmax' que entrega una distribución de probabilidad útil para determinar el valor más relevante en la salida de la Red Neuronal.

Se debe modificar el path (si no se ha hecho aun) y dar un nombre al modelo (modificar nombre_modelo al final de la arquitectura de la red). Ejecutando neural.py en el entorno de gym-duckietown (con librería TensorFlow instalada), se debería apreciar en consola como carga el modelo, entrenado con una cantidad de epocas por defecto. Este programa llama a reader.py, que carga los datos a la memoria RAM para ser utilizados en su conjunto. Cuando termine de cargar, se verá el contador epoch 7/7 y un aviso de que el modelo ya esta listo. Ahora tendrá su modelo en la carpeta models listo para ser usado. **Es posible que neural.py no cargue completamente si su ordenador no tiene las características suficientes para correr su modelo. Por esto también se recomienda ejecutar neural.py junto con reader.py en un Google Colab con GPU activada, trasladando las carpetas de frames, models y el archivo de texto vel.txt.**

Modificar nuevamente el path de autoduck.py y cambiar el nombre al modelo que se creó. Si todo sale como fue estipulado, al ejecutar autoduck.py se deberían ver los resultados de su modelo, generando la conducción autónoma del duckiebot. 

Se ha implementado un modelo con 20.000 imagenes para el uso público y para probar el modelo con un conjunto pre-entrenado. Como ya se mencionó, el link está en models y podrá descargar de ahí este modelo. 
