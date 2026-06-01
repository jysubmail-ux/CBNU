'''
▪ 영상을 흑백으로 변환하고 화면에 출력하시오
▪ 흑백으로 변환된 영상에 Histogram Equalization을 적용하고 출력하시오.
▪ 흑백으로 변환된 영상에 Gamma Correction을 적용하고 출력하시오.
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


grey = cv2.imread('Lena.png', 0)
cv2.imshow('original grey', grey)
cv2.waitKey()


hist, bins = np.histogram(grey, 256, [0,255])
plt.fill(hist)
plt.xlabel('pixel value')
plt.show()

grey_eq = cv2.equalizeHist(grey) # 이미지를 equalizeHist
hist, bins = np.histogram(grey_eq, 256, [0,255]) # 히스토그램 생성
plt.fill_between(range(256), hist, 0) # 히스토그램 채워주는 코드
plt.xlabel('pixel value')
plt.show()

cv2.imshow('equalized gray', grey_eq)
cv2.waitKey()

color = cv2.imread("Lena.png")
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

hsv[..., 2] = cv2.equalizeHist(hsv[...,2])
color_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('original color', color)
cv2.imshow('equalized color', color_eq)

cv2.waitKey()
cv2.destroyAllWindows()