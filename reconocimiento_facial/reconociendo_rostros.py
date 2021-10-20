import cv2
import os
import imutils
import copy

personName = "Brian"
dataPath = "/Users/brianqp/PycharmProjects/proyecto_ia/reconocimiento_facial/data"
personPath = dataPath + "/" + personName

if not os.path.exists(personPath):
    print("Carpeta creada ", personPath)
    os.makedirs(personName)

cap = cv2.VideoCapture(0)
faceClassif = cv2.CascadeClassifier(".//data//haarcascade//haarcascade_frontalface_default.xml")
count = 0

while True:
    ret, frame = cap.read()
    if ret == False: break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(faces, (x,y),(x+w,y+h),color="green", 2)
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150,150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro)
        count += 1

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27 or count >= 300:
        break

    cap.release()
    cv2.destroyWindow()