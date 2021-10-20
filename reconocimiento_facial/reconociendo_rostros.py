import cv2
import os
import imutils
faceClassif = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
image = cv2.imread("gente.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

faceClassif =cv2.CascadeClassifier(".//data//haarcascade//haarcascade_frontalface_default.xml")

faces = faceClassif.detectMultiScale(gray, scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(30,30),
                                     maxSize=(200,200))

for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0))
cv2.imshow("image", image)
cv2.waitKey()
cv2.destroyWindow()