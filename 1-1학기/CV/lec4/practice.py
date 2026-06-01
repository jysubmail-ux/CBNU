import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 이미지 읽기
# -------------------------------
image = cv2.imread('Lena.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# -------------------------------
# (1) Unsharp Mask (예시 코드 방식)
# -------------------------------
KSIZE = 11
ALPHA = 1.5

kernel = cv2.getGaussianKernel(KSIZE, 0)
kernel = -ALPHA * kernel @ kernel.T
kernel[KSIZE//2, KSIZE//2] += 1 + ALPHA

unsharp = cv2.filter2D(gray, -1, kernel)



# -------------------------------
# (2) Sobel Filter
# -------------------------------
sobel_x = cv2.Sobel(unsharp, cv2.CV_64F, 1, 0, ksize=3) # unsharp 이미지를 CV_64F 정밀도로 x축 미분, kernal 사이즈는 3
sobel_y = cv2.Sobel(unsharp, cv2.CV_64F, 0, 1, ksize=3) # unsharp 이미지를 CV_64F 정밀도로 y축 미분, kernal 사이즈는 3

sobel = cv2.magnitude(sobel_x, sobel_y) # Gradient 합치는 과정
sobel = np.uint8(np.clip(sobel, 0, 255)) # 값을 0 ~ 255로 설정

# -------------------------------
# (3) Gabor Filter
# -------------------------------
kernel_gabor = cv2.getGaborKernel((21,21), 5, 1, 10, 1, 0, cv2.CV_32F)
kernel_gabor /= math.sqrt((kernel_gabor * kernel_gabor).sum())
gabor = cv2.filter2D(unsharp, cv2.CV_8UC3, kernel_gabor)

# -------------------------------
# (4) Threshold + Trackbar
# -------------------------------
def nothing(x):
    pass

cv2.namedWindow("Threshold Compare")
cv2.createTrackbar("TH", "Threshold Compare", 50, 255, nothing)

while True:
    th = cv2.getTrackbarPos("TH", "Threshold Compare")

    _, sobel_th = cv2.threshold(sobel, th, 255, cv2.THRESH_BINARY)
    _, gabor_th = cv2.threshold(gabor, th, 255, cv2.THRESH_BINARY)

    diff = cv2.absdiff(sobel_th, gabor_th)

    cv2.imshow("Threshold Compare", diff)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyWindow("Threshold Compare")

# -------------------------------
# (5) Opening & Closing
# -------------------------------
kernel_m = np.ones((5,5), np.uint8)

opening = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel_m)
closing = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel_m)



# -------------------------------
# (6) Discrete Fourier Transform
# -------------------------------
image2 = cv2.imread('Lena.png', 0).astype(np.float32) / 255
fft = cv2.dft(image2, flags=cv2.DFT_COMPLEX_OUTPUT)

shifted = np.fft.fftshift(fft, axes=[0,1])
magnitude = cv2.magnitude(shifted[:,:,0], shifted[:,:,1])
magnitude = np.log(magnitude)

restored = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

# -------------------------------
# 결과 출력
# -------------------------------
plt.figure(figsize=(10,8))

plt.subplot(331)
plt.title("Original")
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(332)
plt.title("Unsharp")
plt.imshow(unsharp, cmap='gray')
plt.axis('off')

plt.subplot(333)
plt.title("Sobel")
plt.imshow(sobel, cmap='gray')
plt.axis('off')

plt.subplot(334)
plt.title("Gabor")
plt.imshow(gabor, cmap='gray')
plt.axis('off')

plt.subplot(335)
plt.title("Opening")
plt.imshow(opening, cmap='gray')
plt.axis('off')

plt.subplot(336)
plt.title("Closing")
plt.imshow(closing, cmap='gray')
plt.axis('off')

plt.subplot(337)
plt.title("DFT")
plt.imshow(magnitude, cmap='gray')
plt.axis('off')

plt.subplot(338)
plt.title("restored")
plt.imshow(restored, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
