import cv2, numpy as np

image = cv2.imread('Lena.png').astype(np.float32) / 255
#
# print('shape:', image.shape)
# print('Data type : ', image.dtype)
# cv2.imshow("original image", image)
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print('Converted to grayscale:')
# print('shape:', gray.shape)
# print('Data type : ', gray.dtype)
# cv2.imshow("gray-scale image", gray)
#
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# print('Converted to HSV')
# print('shape:', hsv.shape)
# print('Data type : ', hsv.dtype)
# cv2.imshow("from_hsv image", hsv)
# cv2.waitKey()
# cv2.destroyAllWindows()

#실습1 Lena 이미지를 컬러영상으로 읽고 화면에 출력하시오.
# cv2.imshow("original image", image)
# cv2.waitKey()

#실습2 영상을 흑백으로 변환하고 화면에 출력하시오
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print('Converted to grayscale:')

cv2.imshow("gray-scale image", gray)
cv2.waitKey()
