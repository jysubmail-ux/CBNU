import cv2
import numpy as np
import random

img = np.full((512,512,3), 255, np.uint8)

axes = (int(256*random.uniform(0,1)), int(256*random.uniform(0,1)))
angle = int(180*random.uniform(0,1))
center = (256,256)

pts = cv2.ellipse2Poly(center, axes, angle, 0, 360, 1)
pts += np.random.uniform(-10, 10, pts.shape).astype(np.int32)

cv2.ellipse(img, center, axes, angle, 0, 360, (0,255,0), 3)

for pt in pts:
    cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0,0,255))

cv2.imshow('Fit ellipse', img)
cv2.waitKey()
cv2.destroyAllWindows()

ellipse = cv2.fitEllipse(pts)
cv2.ellipse(img, ellipse, (0,0,0), 3)

cv2.imshow('Fit ellipse', img)
cv2.waitKey()
cv2.destroyAllWindows()

img = np.full((512, 512, 3), 255, np.uint8)


pts = np.arange(512).reshape(-1, 1)
pts = np.hstack((pts, pts))


pts = pts + np.random.uniform(-10, 10, pts.shape)
pts = pts.astype(np.float32)  # fitLine은 float 필요

cv2.line(img, (0, 0), (511, 511), (0, 255, 0), 3)


for pt in pts:
    cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

cv2.imshow('Fit line - original', img)
cv2.waitKey()
cv2.destroyAllWindows()

vx, vy, x, y = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)


y0 = int(-x * vy / vx + y)
y1 = int((511 - x) * vy / vx + y)


img2 = np.full((512, 512, 3), 255, np.uint8)

for pt in pts:
    cv2.circle(img2, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

cv2.line(img2, (0, y0), (511, y1), (255, 0, 0), 3)

cv2.imshow('Fit line - result', img2)
cv2.waitKey()
cv2.destroyAllWindows()