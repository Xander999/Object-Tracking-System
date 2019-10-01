import cv2
import numpy as np

cam = cv2.VideoCapture(0)
name = input('Enter name :')
name = name + '.jpg'
while True:
    _, frame = cam.read()

    key = cv2.waitKey(1)
    cv2.imshow("frame", frame)
    if key == 27:
        cv2.imwrite(name, frame)
        break

cam.release()
cv2.destroyAllWindows()
