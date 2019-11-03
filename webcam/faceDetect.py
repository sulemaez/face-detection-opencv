import cv2
import sys

face = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
frame = cv2.imread('./rihanna.jpg')
video_capture = cv2.VideoCapture(0)

while True:
      # Capture frame-by-frame
    ret, frame = video_capture.read()

    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(grey,scaleFactor=1.05,minNeighbors=5)

    for x,y,w,h in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)


    cv2.imshow('Video',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()




