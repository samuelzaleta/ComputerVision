import time

import cv2
from joblib import load
from sklearn.pipeline import Pipeline

import PoseModule as pm
import numpy as np

from AttributerRemover import AttributerRemover
from turtle import *

height = 60
width = 60
screen = Screen()
screen.screensize(width, height)
sam =Turtle()
sam.color('black')
sam.setheading(90)
sam.turtlesize(stretch_wid=3)
model = load('model.sav')
cap = cv2.VideoCapture(0)
pTime = 0
detector = pm.PoseDetector()
contador = 0
while True:
    success, img = cap.read()
    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img = detector.find_pose(img, False)
    lmList = detector.find_position(img, False)

    # Analizar solo un punto
    if len(lmList) != 0:
        angleBrazoDerecho = detector.find_angle(img, 12, 14, 16)
        angleHombroDerecho = detector.find_angle(img, 24, 12, 14)
        angleHombroHombroManoDer = detector.find_angle(img, 12, 16, 11, False)
        angleCaderaHombroDerManoIz = detector.find_angle(img, 24, 15, 12, False)
        angleBrazoIzquierdo = detector.find_angle(img, 11, 13, 15)
        angleHombroIzquierdo = detector.find_angle(img, 23, 11, 13)
        angleHombroHombroManIzq = detector.find_angle(img, 12, 15, 11, False)
        angleCaderaHombroIzqManoDer = detector.find_angle(img, 23, 16, 11, False)

        X = np.c_[angleBrazoDerecho, angleHombroDerecho, angleHombroHombroManoDer, angleCaderaHombroDerManoIz,
                  angleBrazoIzquierdo, angleHombroIzquierdo, angleHombroHombroManIzq, angleCaderaHombroIzqManoDer]
        X = X / 360

        pipeline = Pipeline([
            ('attributes', AttributerRemover(remove_hombro_hombro_der=True)),
        ])
        X = pipeline.transform(X)
        y = model.predict(X)

        '''
        # Brazo derecho
        angleBrazoDerecho = detector.find_angle(img, 12, 14, 16)
        porcentajeBrazoDerecho = np.interp(angleBrazoDerecho, (180, 340), (0, 100))
        # Hombro derecho
        angleHombroDerecho = detector.find_angle(img, 24, 12, 14)
        porcentajeHombroDerecho = np.interp(angleHombroDerecho, (15, 100), (0, 100))
        # check move rigth
        # Direccion Derecha
        if porcentajeHombroDerecho > 65 and (23 > porcentajeBrazoDerecho > 0):
            cv2.putText(img, "Derecha", (70, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 0), 3)

        if (45 < porcentajeHombroDerecho < 70) and (55 < porcentajeBrazoDerecho < 75):
            cv2.putText(img, "Enfrente", (70, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 0), 3)
        # Direccion Derecha
        if porcentajeHombroDerecho > 95 and porcentajeBrazoDerecho > 70:
            cv2.putText(img, "Atras", (70, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 0), 3)
        # Brazo Izquierdo
        angleBrazoIzquierdo = detector.find_angle(img, 11, 13, 15)
        porcentajeBrazoIzquierdo = np.interp(angleBrazoIzquierdo, (140, 200), (0, 100))
        # Hombro Izquierdo
        angleHombroIzquierdo = detector.find_angle(img, 23, 11, 13)
        porcentajeHombroIzquierdo = np.interp(angleHombroIzquierdo, (340, 270), (0, 100))
        # Dirección Izquierda
        print('Hombro derecho', porcentajeHombroDerecho, 'Brazo derecho', porcentajeBrazoDerecho)
        if porcentajeHombroIzquierdo > 70 and porcentajeBrazoIzquierdo < 50:
            cv2.putText(img, "Izquierda", (70, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 0), 3)
        '''
        print(y[0])
        cv2.putText(img, y[0], (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        if y[0] == 'Enfrente':
            contador +=1
            if contador >48:
                contador = 0
                sam.setheading(90)
                sam.forward(50)
        elif y[0] == 'Izquierda':
            contador += 1
            if contador > 48:
                contador = 0
                sam.setheading(180)
                sam.forward(50)
        elif y[0] == 'Derecha':
            contador += 1
            if contador > 48:
                contador = 0
                sam.setheading(360)
                sam.forward(50)
        elif y[0] == 'Atras':
            contador += 1
            if contador > 48:
                contador = 0
                sam.setheading(270)
                sam.forward(50)
        elif y[0] == 'Inicio':
            contador += 1
            if contador > 150:
                sam.reset()
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    #aproximadamente va 12-14 fps
    pTime = cTime
    cv2.imshow("Image", img)
    cv2.waitKey(1)

