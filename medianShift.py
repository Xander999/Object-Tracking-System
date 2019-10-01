import cv2
import  numpy as np

first_frame=cv2.imread("greenCard.jpg")
#cv2.imshow("Real",first_frame)
'''
#redCard
x=261
y=121
width=166
height=243
'''
#greenCard
x=231
y=169
width=221
height=117
'''
#blueCard
x=310
y=148
width=180
height=212
'''
roi=first_frame[y:y+height,x:x+width]
#cv2.imshow("Region",roi)

hsv_roi=cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist=cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
roi_hist=cv2.normalize(roi_hist, roi_hist,0,255,cv2.NORM_MINMAX)

term=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 10, 1)

cam=cv2.VideoCapture(0)

while True:
    _,frame=cam.read()
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask=cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

    cv2.imshow("mask",mask)
    _,track=cv2.meanShift(mask,(x,y,width,height),term)
    x,y,w,h=track
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("frame",frame)

    key=cv2.waitKey(1)
    if key==27:
        break

cam.release()
cv2.destroyAllWindows()