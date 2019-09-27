# -*- coding: UTF-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

def redim(img, larg): 
    alt = int(img.shape[0]/img.shape[1]*larg)
    img = cv2.resize(img, (larg, alt), interpolation = cv2.INTER_AREA)
    return img

df = cv2.CascadeClassifier("modelo/haarcascade_frontalface_default.xml")

video = cv2.VideoCapture("video/bebe.mp4")
count = 0

while True:
    (sucesso, frame) = video.read()
    if not sucesso:
        break

    frame = redim(frame, 320)
    frame_pb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = df.detectMultiScale(frame_pb, scaleFactor = 1.1, minNeighbors=3, minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)
    frame_temp = frame.copy()

    hist_todos = cv2.calcHist([frame_temp],[0],None,[256],[0,256])
    plt.plot(hist_todos)

    plt.savefig("Hist/histograma"+str(count)+".png")
    plt.clf()

    for (x, y, lar, alt) in faces:
        imgS = cv2.rectangle(frame_temp, (x, y), (x + lar, y + alt), (0,255, 255), 2)

        rec = imgS[y:y+alt,x:x+lar]
        cv2.imwrite("Pessoas/faceId"+str(count)+".png",rec)

        hist_full = cv2.calcHist([imgS],[0],None,[256],[0,256])
        plt.plot(hist_full)

        plt.savefig("HistFace/histogramaFace"+str(count)+".png")
        plt.clf()

    count += 1

    cv2.imshow("Faces...", redim(frame_temp, 640))

    if cv2.waitKey(1) & 0xFF == ord("s"):
        break

video.release()
cv2.destroyAllWindows()