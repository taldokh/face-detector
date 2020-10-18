import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('avengers3.jpg')

gratscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(gratscaled_img)

for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('somthing', img)
cv2.waitKey()

print('Code Completed')

