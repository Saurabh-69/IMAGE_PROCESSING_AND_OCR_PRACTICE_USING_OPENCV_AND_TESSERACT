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


#BITWISE OPERATIONS

blank = np.zeros((400,400), dtype='uint8')

rectangle = cv2.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
circle = cv2.circle(blank.copy(), (200,200), 200, 255, -1 )

cv2.imshow("circle",circle)
cv2.imshow("recyt", rectangle) 


bitwise_and = cv2.bitwise_and(rectangle, circle)
cv2.imshow("rectandcircle", bitwise_and)


bitwise_or = cv2.bitwise_or(rectangle, circle)
cv2.imshow("rectorcircle", bitwise_or)


bitwise_xor  = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("rectxorcircle", bitwise_xor)

bitwise_not = cv2.bitwise_not(rectangle)
cv2.imshow("rectnotcir", bitwise_not)

# MASKING

blank = np.zeros(image.shape[0:2], dtype="uint8")
cv2.imshow("blank", blank) 

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayed", gray)


mask = cv2.circle(blank, (image.shape[1]//2, image.shape[0]//2), 100, 255, -1)
cv2.imshow("masked", mask)

masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("masked1", masked)

cv2.waitKey(10000)
cv2.destroyAllWindows()
