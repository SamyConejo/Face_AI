# Face AI

Proyecto Integrador
@autor: SamyConejo

## Getting Started

Este proyecto usa Google Flutter. Para reconstruir la aplicacion se debe descargar o clonar el repositorio y ejecutar un "pub get" en la consola del IDE. 
Las versiones requeridas para cada libreria se encuentran en el archivo pubspeck.yaml. El deployment y configuración se dio sobre iOS 16 con iPhone 12 Pro y con macOS Ventura 13.

Para la parte de entrenamiento del modelo los scripts estan listos para correr. Solo se debe especificar el path hacia las directorios donde se encuentran los folders de imágenes. Los datasets se pueden descargar de los enlaces anexados. 

## Folder Entrenamiento 
Contiene los scripts de Python para la creación del modelo.
  - Extraccion de frames en un video y detección de rostros (rostros.ipynb).
  - Entrenamiento del modelo con Transferencia de Aprendizaje (face_recognition.ipynb).
  - Entrenamiento del modelo con Transferencia de Aprendizaje y Stratified Cross Validation (face_recognition_cv.py).
  - Archivo .csv para entrenamiento con Stratified Cross Validation (images.csv).
  - Modelos exportados (.h5 y .tflite) para el mejor modelo aplicando SCV.
  
## Data Split Clásico
Para la primera experimentación se usó directamente el estructura de directorios adjunta en el primer enlace. 

## Stratified Cross Validation
Para mejorar la generalización del modelo y siguiendo la norma ISO 23053:2022 se configuró el entrenamiento aplicando Stratified Cross Validation con un valor de k = 10. Para esta experimentación los folders de train y validation se combinaron en una sola carpeta llamada data. Cada imagen se renombró de forma tal que se creó un archivo csv que mapea el path de la imagen con la clase a la que pertece. Sobre este archivo csv se aplica la estrategia de SCV por lo que el ImageDataGenerator solo necesita saber el path de las imagenes seleccionadas para cada fold y no carga en memoria todas la imagenes innecesariamente. 

## Base de datos
  - La base de datos (data split clásico) se puede descargar en el siguiente enlace: 
  https://www.icloud.com/iclouddrive/0b1dAWS2KYOdoUBcGRjtXpNPw#dataset  
  
   - La base de datos (Stratified Cross Validation) se puede descargar en el siguiente enlace: 
  https://www.icloud.com/iclouddrive/0539IrlHhe-kNkwXxp72mfWXg#dataset_cv

