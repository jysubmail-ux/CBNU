'''
영상을 HSV 컬러 스페이스로 변환하고 각각을 화면에 출력하시오. (각값을 0부터 255로 정규화)
▪ H 채널에 대해서 Median Filter를 적용하고 출력하시오.
▪ S 채널에 대해서 Gaussian Filter를 적용하고 출력하시오.
▪ V 채널에 대해서 Bilateral Filter를 적용하고 출력하시오.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Lena.png').astype(np.float32) / 255

# 노이즈 추가
noised = (image + 0.2 * np.random.rand(*image.shape).astype(np.float32))
noised = noised.clip(0,1)

plt.figure()
plt.imshow(noised[:, :, [2,1,0]])
plt.title("Noised Image")
plt.axis('off')
plt.show()


# Gaussian Blur
gauss_blur = cv2.GaussianBlur(noised, (7,7), 0)

plt.figure()
plt.imshow(gauss_blur[:, :, [2,1,0]])
plt.title("Gaussian Blur")
plt.axis('off')
plt.show()


# Median Blur
median_blur = cv2.medianBlur((noised * 255).astype(np.uint8), 7)

plt.figure()
plt.imshow(median_blur[:, :, [2,1,0]])
plt.title("Median Filter")
plt.axis('off')
plt.show()


# Bilateral Filter
bilat = cv2.bilateralFilter(noised, -1, 0.3, 10)

plt.figure()
plt.imshow(bilat[:, :, [2,1,0]])
plt.title("Bilateral Filter")
plt.axis('off')
plt.show()