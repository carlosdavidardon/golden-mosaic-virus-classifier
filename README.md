# golden-mosaic-virus-classifier
Colección de programas para preprocesamiento de imágenes del dataset, entrenamiento de redes neuronales por transferencia de aprendizaje y análisis posteriores.


## Descripción de los programas

* confusion_matrix.py: calcula la matriz de confusión de dos redes neuronales sobre el mismo conjunto de pruebas.
* converToLite.py: convierte una red neuronal en formato HDF5 (.h5) a tflite para TensorFlow lite.
* mcnemar_test.py: calcula una matriz de contingencia y el test de McNemar para comparar dos redes neuronales.
* predict.py: usa una red neuronal pre-entrenada para realizar la inferencia de clasificación de una imagen.
* scale_image.py: reduce el tamaño de las imágenes de un conjunto de datos y las coloca en otro directorio.
* split_dataset.py: toma un conjunto de datos y aleatoriamente mueve una porción a otro directorio.
* test_model.py: calcula la exactitud de una red neuronal sobre un conjunto de datos.
* train_net_keras.py: entrenamiento de redes neuronales (InceptionV3 y ResNet50V2) por medio de transferencia de aprendizaje con todas las capas.
* train_net_keras_lower.py: entrenamiento de redes neuronales (InceptionV3 y ResNet50V2) por medio de transferencia de aprendizaje a partir de capas intermedias.
