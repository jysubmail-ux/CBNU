import cv2
import numpy as np
import random

# 이미지 불러오기 (흑백)
image = cv2.imread('Lena.png', 0)

# ---------------------------
# (1) Otsu Thresholding
# ---------------------------
_, thresh = cv2.threshold(
    image, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

cv2.imshow("Otsu Threshold", thresh)


# ---------------------------
# (2) External / Internal Contour
# ---------------------------
result = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

if len(result) == 3:
    _, contours, hierarchy = result
else:
    contours, hierarchy = result

# 외부 / 내부 이미지
external = np.zeros_like(image)
internal = np.zeros_like(image)

for i in range(len(contours)):
    # 부모 없음 → 외부 contour
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(external, contours, i, 255, -1)
    else:
        cv2.drawContours(internal, contours, i, 255, -1)

cv2.imshow("External Contour", external)
cv2.imshow("Internal Contour", internal)


# ---------------------------
# (3) Connected Component + 랜덤 5개
# ---------------------------
num_labels, labels = cv2.connectedComponents(thresh)

def show_random_components():
    output = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # 0번은 background라 제외
    chosen = random.sample(range(1, num_labels), min(5, num_labels-1))

    for label in chosen:
        mask = (labels == label)
        color = np.random.randint(0, 255, size=3)

        output[mask] = color

    cv2.imshow("Random 5 Components", output)


# ---------------------------
# (4) Distance Transform
# ---------------------------
dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

# 보기 좋게 normalize
dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
dist_norm = dist_norm.astype(np.uint8)

cv2.imshow("Distance Transform", dist_norm)


# ---------------------------
# 키 입력 처리
# ---------------------------
print("SPACE 누르면 랜덤 컴포넌트 5개 표시 / ESC 종료")

while True:
    key = cv2.waitKey(0)

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        show_random_components()

cv2.destroyAllWindows()