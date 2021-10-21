import cv2
import os
import imutils
dataPath = "/Users/brianqp/PycharmProjects/proyecto_ia/reconocimiento_facial/data"
peopleList = os.listdir(dataPath)
print(peopleList)

labels = []
facesData = [] # Se crean labels y facesData para almacenar la imformacion de cada persona y guardarla con un ID
label = 0

for nameDir in peopleList: # Recorre todas las carpetas con nombre de personas
    personPath = dataPath + '/' + nameDir
    print("Leyendo las imagenes")
    for fileName in os.listdir(personPath): # Recorre cada archivo del directorio y le asigna un label
        print('Rostro: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath + '/' + fileName, 0))
        image = cv2.imread(personPath + '/' + fileName,0)
    label += 1

print(len(labels))