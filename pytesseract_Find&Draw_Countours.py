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


#FINDING AND DRAWING COUNTOURS

blank = np.zeros(image.shape, dtype='uint8')
cv2.imshow("blank", blank)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)


canny = cv2.Canny(image, 125, 175)
cv2.imshow("canny edges", canny)


blur = cv2.GaussianBlur(gray, (1,1), cv2.BORDER_DEFAULT)
cv2.imshow("blur", blur)


ret, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
cv2.imshow("gray", thresh)


countours, hierarchies = cv2.findContours(thresh ,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print(f"{len(countours)} countour (s) found! ")


cv2.drawContours(blank, countours, -1, (0,0,255), 1)
cv2.imshow ("countors drawn", blank )


# COUNTOURS
thresh = cv2.threshold(blur,0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
print(cv2.imwrite("temp/thresh.png",thresh))


kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(3,13))
print(cv2.imwrite("temp/index_kernal.png",kernal))

dialate = cv2.dilate(thresh, kernal, iterations=1 )
print(cv2.imwrite("temp/index_dialate.png",dialate))


cnts = cv2.findContours(dialate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]


cnts = sorted(cnts, key=lambda x:cv2.boundingRect(x)[0])

for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    # Calculate bottom-right corner coordinate
    bottom_right = (x + w, y + h)
    # if h>100 and w > 20:
        # roi = image[y:y+h, x:x+h] 
    # cv2.imwrite("temp/index_bbox_new.png",roi)
    cv2.rectangle(image,(x,y), bottom_right, (36,255,12),2)
    ocr_result = pytesseract.image_to_string(image)
    print(ocr_result)

print(cv2.imwrite("temp/index_bbox.png",image))

cv2.waitKey(10000)
cv2.destroyAllWindows()
