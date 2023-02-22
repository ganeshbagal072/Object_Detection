import cv2

from random import randrange

pre_trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#img = cv2.imread('facec.jpg')

webcam = cv2.VideoCapture('IMG_0447.mov')

while True:

    successful_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = pre_trained_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
    
    cv2.imshow('Ganesh Bagal Face Detector', frame)
    Key = cv2.waitKey(1)

    if Key==81 or Key==113:
        break

webcam.release()


