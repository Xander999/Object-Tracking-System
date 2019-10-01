import cv2
import  numpy as np

first_frame=cv2.imread("redCard.jpg")
#redCard
x=261
y=121
width=166
height=243
roi=first_frame[y:y+height,x:x+width]

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
    rem, track=cv2.CamShift(mask,(x,y,width,height),term)
    pts=cv2.boxPoints(rem)
    pts=np.int0(pts)
    cv2.polylines(frame,[pts],True,[255,0,0],2)
    cv2.imshow("frame",frame)

    key=cv2.waitKey(1)
    if key==27:
        break

cam.release()
cv2.destroyAllWindows()