import matplotlib.pyplot as plt
import pytesseract
import numpy as np
import cv2

image = cv2.imread(r"C:\Users\saurabh.kale\Downloads\SQL_QUERY_GENERATOR_TWO\SQL_QUERY_GENERATOR\SQL_QUERY_GENERATOR\cat_image.jfif")
# image2 = cv2.imread(r"C:\Users\saurabh.kale\Downloads\SQL_QUERY_GENERATOR_TWO\SQL_QUERY_GENERATOR\SQL_QUERY_GENERATOR\mumbai.jpg")
base_image = image.copy()
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\saurabh.kale\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
 
#DISPLAYTING THE ORIGINAL IMAGE
cv2.imshow("OG",image)

#GRAYSCALING THE IMAGE
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAYED IMAGE", gray)



# IMPORTANT FUNCTION/ OPERATIONS IN OPENCV

#GRAYSCALING THE IMAGE
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(cv2.imwrite("temp/gray.png", gray))
cv2.imshow("OG IMAGE",base_image)
cv2.imshow("gray",gray)

#BLURING THE IMAGE
blur = cv2.GaussianBlur(gray, (9,9),3)
print(cv2.imwrite("temp/blur.png", gray))
cv2.imshow("blur",blur)


#EDGE CASCADE 
cany =  cv2.Canny(image,125,175)
cv2.imwrite("temp/cany.png",cany)
cv2.imshow("canny", cany)


# HSV THE IMAGE
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv", hsv)

# LAB THE IMAGE
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("lab", lab)


#DIALTE
dialted = cv2.dilate(cany, (3,3), iterations=1)
cv2.imshow("dialte.png",dialted)


#ERODING THE IMAGE
eroded = cv2.erode(dialted,(3,3), iterations=1)
cv2.imshow("eroded.png",dialted)


#RESIZE AND CROP
resized = cv2.resize(image, (500,500))
cv2.imshow("resized", resized)


#CROPPING
croped = image[50:500,200:500]
cv2.imshow("cropped",croped)


#TRANSLATION
def translate(image,x,y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (image.shape[1], image.shape[0] )
    return cv2.warpAffine(image,transMat,dimensions)

translated = translate(image, 100,100)
cv2.imshow("translated", translated)


#ROTATION
def rotation(image,angle, rotPoint=None):
    (height,width) = image.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2)
    
    rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)
    return cv2.warpAffine(image, rotMat, dimensions)

rotated = rotation(image, 45, None)
cv2.imshow("rotated", rotated)

rotated_rotated = rotation(rotated, 45)
cv2.imshow("rotated rotated", rotated_rotated)


# FLIPPING

flip = cv2.flip(image, 0)
cv2.imshow("flip", flip)

flip1 = cv2.flip(image, -1)
cv2.imshow("flip1", flip1)

b,g,r = cv2.split(image)

cv2.imshow("blue", b)
cv2.imshow("red", r)
cv2.imshow("green", g)

# merge_image = cv2.merge([b,g,r])
# cv2.imshow("merged", merge_image)

blank = np.zeros(image.shape[:2], dtype='uint8') 

blue = cv2.merge([b,blank,blank])
green = cv2.merge([blank,g,blank])
red = cv2.merge([blank,blank,r])

cv2.imshow("blue",blue)
cv2.imshow("green", green)
cv2.imshow("red", red)



cv2.waitKey(10000)
cv2.destroyAllWindows()
