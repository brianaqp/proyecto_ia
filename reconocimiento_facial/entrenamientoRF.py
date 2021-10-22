import cv2
import os
import imutils
import numpy as np
# dataPath = "/Users/brianqp/PycharmProjects/proyecto_ia/reconocimiento_facial/data/personas" # Mac
dataPath = "C:/Users/brian/PycharmProjects/proyecto_ia/reconocimiento_facial/data/personas" # Windows
peopleList = os.listdir(dataPath)
print(peopleList)

labels = []
facesData = [] # Se crean labels y facesData para almacenar la imformacion de cada persona y guardarla con un ID
label = 0

for nameDir in peopleList: # Recorre todas las carpetas con nombre de personas
    if nameDir == '.DS_Store':
        continue # Evita tomar la carpeta que necesito en mi mac
    personPath = dataPath + '/' + nameDir
    print("Leyendo las imagenes")
    for fileName in os.listdir(personPath): # Recorre cada archivo del directorio y le asigna un label
        print('Rostro: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath + '/' + fileName, 0))
        image = cv2.imread(personPath + '/' + fileName,0)
    label += 1

face_recognizer = cv2.face.EigenFaceRecognizer_create()
# Entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# almacenando el modulo obtenido
face_recognizer.write('modeloEigenFaces.xml')
print('modulo entrenado')
