import cv2
import numpy as np
import random

img = cv2.imread('BnW.png', cv2.IMREAD_GRAYSCALE)

result = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if len(result) == 3:
    _, contours, hierarchy = result
else:
    contours, hierarchy = result

color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(color, contours, -1, (0, 255, 0), 3)

cv2.imshow('contours', color)
cv2.waitKey()
cv2.destroyAllWindows()

contour = contours[0]

print('Area of contour is %.2f' % cv2.contourArea(contour))
print('Signed area of contour is %.2f' % cv2.contourArea(contour, True))
print('Signed area of contour is %.2f' % cv2.contourArea(contour[::-1], True))

print('Length of closed contour is %.2f' % cv2.arcLength(contour, True))
print('Length of open contour is %.2f' % cv2.arcLength(contour, False))

hull = cv2.convexHull(contour)
cv2.drawContours(color, [hull], -1, (0, 0, 255), 3)

cv2.imshow('contours', color)
cv2.waitKey()
cv2.destroyAllWindows()

print('Convex status of contour is %s' % cv2.isContourConvex(contour))
print('Convex status of its hull is %s' % cv2.isContourConvex(hull))

cv2.namedWindow('contours')

img_draw = np.copy(color)

def trackbar_callback(value):
    global img_draw
    epsilon = value * cv2.arcLength(contour, True) * 0.1 / 255
    approx = cv2.approxPolyDP(contour, epsilon, True)

    img_draw = np.copy(color)
    cv2.drawContours(img_draw, [approx], -1, (255, 0, 255), 3)

cv2.createTrackbar('Epsilon', 'contours', 1, 255, trackbar_callback)

while True:
    cv2.imshow('contours', img_draw)
    key = cv2.waitKey(3)
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()