import cv2
import sys

face = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
img = cv2.imread('g.jpg')

grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face.detectMultiScale(grey,scaleFactor=1.05,minNeighbors=5)

for x,y,w,h in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

cv2.imshow("rihanna",img)
if cv2.waitKey(0) == ord("q"):
    cv2.destroyAllWindows()


print(type(faces))
print(faces.shape)


