'''
▪ 영상을 흑백으로 변환하고 화면에 출력하시오
▪ 흑백으로 변환된 영상에 Histogram Equalization을 적용하고 출력하시오.
▪ 흑백으로 변환된 영상에 Gamma Correction을 적용하고 출력하시오.
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('Lena.png', 0).astype(np.float32) / 255

gamma = 0.5
corrected_img = np.power(image, gamma)


cv2.imshow('original image', image)
cv2.imshow('corrected image', corrected_img)
cv2.waitKey()
