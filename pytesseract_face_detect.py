import matplotlib.pyplot as plt
import pytesseract
import numpy as np
import cv2

# READING THE IMAGE
image = cv2.imread(r"C:\Users\saurabh.kale\Downloads\SQL_QUERY_GENERATOR_TWO\SQL_QUERY_GENERATOR\SQL_QUERY_GENERATOR\group3.jpg")
base_image = image.copy()
image10 = cv2.imread(r"C:\Users\saurabh.kale\Downloads\SQL_QUERY_GENERATOR_TWO\SQL_QUERY_GENERATOR\SQL_QUERY_GENERATOR\pexels-fauxels-3184398.jpg")
image2 = cv2.imread(r"C:\Users\saurabh.kale\Downloads\SQL_QUERY_GENERATOR_TWO\SQL_QUERY_GENERATOR\SQL_QUERY_GENERATOR\pexels-hikaique-109919.jpg")
image3 = cv2.imread(r"C:\Users\saurabh.kale\Downloads\SQL_QUERY_GENERATOR_TWO\SQL_QUERY_GENERATOR\SQL_QUERY_GENERATOR\pexels-jopwell-2422290.jpg")
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\saurabh.kale\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"



# Set the desired width and height
new_width = 600  # Width in pixels
new_height = 600  # Height in pixels
# Resize the image
image1 = cv2.resize(image, (new_width, new_height))
image11 = cv2.resize(image10, (new_width, new_height))
image21 = cv2.resize(image2, (new_width, new_height))
image31 = cv2.resize(image3, (new_width, new_height))

#DISPLAYTING THE ORIGINAL IMAGE AND ORIGINAL IMAGE
# cv2.imshow("OG",image)
cv2.imshow("RESIZED IMAGE", image1)


#GRAYSCALING THE IMAGE
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAYED IMAGE", gray)

gray11 = cv2.cvtColor(image11, cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAYED IMAGE 1", gray11)

gray22 = cv2.cvtColor(image21, cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAYED IMAGE 2 ", gray22)

gray33 = cv2.cvtColor(image31, cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAYED IMAGE 3 ", gray33)


#CANNY IMAGE
# canny = cv2.Canny(image1, 125,175, )
# cv2.imshow("canny image",canny)



# READING THE HARCASCADE XML FILE
haar_cascade = cv2.CascadeClassifier(r"C:\Users\saurabh.kale\Downloads\SQL_QUERY_GENERATOR_TWO\SQL_QUERY_GENERATOR\SQL_QUERY_GENERATOR\haar_face.xml")



#FACE DETECTION
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)

print(f"NUMBER OF FACES FOUND = {len(faces_rect)}")

for (x,y,w,h) in faces_rect:
    cv2.rectangle(image1, (x,y), (x+w, y+h), (0,255,0), thickness=2)
cv2.imshow("detected faces", image1)


faces_rect = haar_cascade.detectMultiScale(gray11, scaleFactor=1.1, minNeighbors=6)
print(f"NUMBER OF FACES FOUND = {len(faces_rect)}")

for (x,y,w,h) in faces_rect:
    cv2.rectangle(image10, (x,y), (x+w, y+h), (0,255,0), thickness=2)
cv2.imshow("detected faces 1", image10)


faces_rect = haar_cascade.detectMultiScale(gray22, scaleFactor=1.1, minNeighbors=1)
print(f"NUMBER OF FACES FOUND = {len(faces_rect)}")

for (x,y,w,h) in faces_rect:
    cv2.rectangle(image21, (x,y), (x+w, y+h), (0,255,0), thickness=2)
cv2.imshow("detected faces 2", image21)


faces_rect = haar_cascade.detectMultiScale(gray33, scaleFactor=1.1, minNeighbors=1)
print(f"NUMBER OF FACES FOUND = {len(faces_rect)}")

for (x,y,w,h) in faces_rect:
    cv2.rectangle(image31, (x,y), (x+w, y+h), (0,255,0), thickness=2)
cv2.imshow("detected faces 3", image31)





















































cv2.waitKey(10000)
cv2.destroyAllWindows()