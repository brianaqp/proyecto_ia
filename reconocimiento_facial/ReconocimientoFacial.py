import cv2
import os
import imutils
import numpy as np

# dataPath = "/Users/brianqp/PycharmProjects/proyecto_ia/reconocimiento_facial/data" # Mac
dataPath = "C:/Users/brian/PycharmProjects/proyecto_ia/reconocimiento_facial/data" # Windows
imagePath = os.listdir(dataPath)
print('imagePath=', imagePath)

face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.read('modeloEigenFaces.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Esto es para usar la camara

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (120, 120), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        cv2.putText(frame, '{}'.format(result),(x,y+5),1,1.3,(0,255,0),2, cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

