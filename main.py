import PoseModule as pm
import os
import cv2

detector = pm.PoseDetector(mode=True)
file = open('dataset.csv', 'w')
file.write('BrazoDerecho,HombroDerecho,HombroHombroManoDerecha,CaderaHombroDerechoManoIzquierda,'
           'BrazoIzquierdo,HombroIzquierdo,HombroHombroManoIzquierda,CaderaHombroIzquierdoManoDerecho,'
           'Posicion\n')
for image_file in os.listdir('./DataSetImages'):
    img = cv2.imread(f'./DataSetImages/{image_file}')
    pose = detector.find_pose(img, False)
    landmarks = detector.find_position(pose, False)

    if len(landmarks) == 0:
        raise RuntimeError(f'El archivo {image_file} no tiene puntos de referencia')

    label = image_file.split('-')[0]
    angleBrazoDerecho = detector.find_angle(img, 12, 14, 16, False)
    angleHombroDerecho = detector.find_angle(img, 24, 12, 14, False)
    angleHombroHombroManoDer = detector.find_angle(img, 12, 16, 11, False)
    angleCaderaHombroDerManoIz = detector.find_angle(img, 24, 15, 12, False)
    angleBrazoIzquierdo = detector.find_angle(img, 11, 13, 15, False)
    angleHombroIzquierdo = detector.find_angle(img, 23, 11, 13, False)
    angleHombroHombroManIzq = detector.find_angle(img, 12, 15, 11, False)
    angleCaderaHombroIzqManoDer = detector.find_angle(img, 23, 16, 11, False)
    file.write(f'{angleBrazoDerecho},{angleHombroDerecho},{angleHombroHombroManoDer},{angleCaderaHombroDerManoIz},'
               f'{angleBrazoIzquierdo},{angleHombroIzquierdo},{angleHombroHombroManIzq},{angleCaderaHombroIzqManoDer},'
               f'{label}\n')
    #cv2.imshow("Image", img)
    #cv2.waitKey(0)
file.close()
