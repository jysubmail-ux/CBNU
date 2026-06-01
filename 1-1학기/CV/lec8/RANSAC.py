import cv2
import numpy as np
import matplotlib.pyplot as plt

img0 = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('Lena_rotated.png', cv2.IMREAD_GRAYSCALE)

detector = cv2.ORB_create(100)

kps0, fea0 = detector.detectAndCompute(img0, None)
kps1, fea1 = detector.detectAndCompute(img1, None)

matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, False)
matches = matcher.match(fea0, fea1)

pts0 = np.float32([kps0[m.queryIdx].pt for m in matches]).reshape(-1, 2) # 원본 이미지 좌표
pts1 = np.float32([kps1[m.trainIdx].pt for m in matches]).reshape(-1, 2) # 변화된 이미지 좌표

# pts0 에서 pts1이 어떻게 된 것인지 계산하는 방법
# cv2.RANSAC을 사용하여 outlier 제거.
# 뒤 3.0의 경우 threshold (거리가 3 이하면 inlier로 판단)
H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)

plt.figure()

plt.subplot(211)
plt.axis('off')
plt.title('all matches')
dbg_img = cv2.drawMatches(img0, kps0, img1, kps1, matches, None)
plt.imshow(dbg_img[:, :, ::-1])

plt.subplot(212)
plt.axis('off')
plt.title('filtered matches')
dbg_img = cv2.drawMatches(img0, kps0, img1, kps1,
    [m for i, m in enumerate(matches) if mask[i]],
    None
)
plt.imshow(dbg_img[:, :, ::-1])

plt.tight_layout()
plt.show()

print(H)
