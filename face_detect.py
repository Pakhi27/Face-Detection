import cv2 as cv
import numpy as np

# haar_cascade-easy to use
# Face detection-detects whether a face is present or not

img = cv.imread("C:\\Users\\singh\\OneDrive\\Pictures\\lady.jpeg")
cv.imshow("Person",img)

# group of people
img_1 = cv.imread("C:\\Users\\singh\\OneDrive\\Pictures\\group.jpeg")

# group of many people
img_2=cv.imread("C:\\Users\\singh\\OneDrive\\Pictures\\people.jpeg")
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("gray person",gray)

# group of people
gray_1=cv.cvtColor(img_1,cv.COLOR_BGR2GRAY)
cv.imshow("gray person",gray_1)
haar_cascade=cv.CascadeClassifier('haar_face.xml')
gray_2=cv.cvtColor(img_2,cv.COLOR_BGR2GRAY)
cv.imshow("gray person",gray_2)

#min neighbors-no of rectangles to ensure it is a face-detect the face and give its coordinates in faces_rect
faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
faces_rect=haar_cascade.detectMultiScale(gray_1,scaleFactor=1.1,minNeighbors=1)
faces_rect=haar_cascade.detectMultiScale(gray_2,scaleFactor=1.1,minNeighbors=3)
print(f'number of faces found={len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img_2,(x,y),(x+w,y+h),(0,255,0),thickness=2)


cv.imshow('detected faces',img_2)

cv.waitKey(0)