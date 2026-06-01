import cv2
import numpy as np
import random

img = cv2.imread('BnW.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("이미지 로드 실패")
    exit()

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
image_to_show = np.copy(color)
measure = True

def mouse_callback(event, x, y, flags, param):
    global contour, image_to_show, measure

    if event == cv2.EVENT_LBUTTONUP:
        distance = cv2.pointPolygonTest(contour, (x, y), measure)

        image_to_show = np.copy(color)

        if distance > 0:
            pt_color = (0, 255, 0)  # 내부
        elif distance < 0:
            pt_color = (0, 0, 255)  # 외부
        else:
            pt_color = (128, 0, 128)  # 경계

        cv2.circle(image_to_show, (x, y), 5, pt_color, -1)

        cv2.putText(
            image_to_show,
            f"{distance:.2f}",
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

cv2.namedWindow('contours')
cv2.setMouseCallback('contours', mouse_callback)

while True:
    cv2.imshow('contours', image_to_show)
    k = cv2.waitKey(1)

    if k == ord('m'):
        measure = not measure
        print("measure mode:", measure)

    elif k == 27:  # ESC
        break

cv2.destroyAllWindows()