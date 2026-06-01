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

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imshow("from_hsv image", hsv)
cv2.waitKey()

h, s, v = cv2.split(hsv)

# =========================
# 1. 0~255 정규화
# =========================
h_norm = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
s_norm = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
v_norm = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# =========================
# 2. 필터 적용
# =========================

# H → Median Filter
h_med = cv2.medianBlur(h_norm, 5)

# S → Gaussian Filter
s_gauss = cv2.GaussianBlur(s_norm, (5, 5), 0)

# V → Bilateral Filter
v_bilat = cv2.bilateralFilter(v_norm, 9, 75, 75)

# =========================
# 3. 출력
# =========================
cv2.imshow("H original", h_norm)
cv2.imshow("H median", h_med)

cv2.imshow("S original", s_norm)
cv2.imshow("S gaussian", s_gauss)

cv2.imshow("V original", v_norm)
cv2.imshow("V bilateral", v_bilat)

cv2.waitKey(0)
cv2.destroyAllWindows()