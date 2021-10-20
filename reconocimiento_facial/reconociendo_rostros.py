import cv2
import os
import imutils
faceClassif = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
image = cv2.imread("gente.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

faceClassif =cv2.CascadeClassifier(".//data//haarcascade//haarcascade_frontalface_default.xml")

faces = faceClassif.detectMultiScale(gray, scaleFactor=1.1, # Variable que afecta que tanto se reducira la imagen
                                     minNeighbors=5,    # Minima cantidad de vecinos que son la misma persona
                                     minSize=(50,50),   # Tamano minimo de los rostros
                                     maxSize=(200,200)) # Tamano maximo de los rostros

for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0))
cv2.imshow("image", image)
cv2.waitKey()
cv2.destroyWindow()