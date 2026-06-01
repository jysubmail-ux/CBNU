import cv2
import numpy as np
from random import randint

# 이미지 로드
img = cv2.imread('Lena.png')
show_img = np.copy(img)

# seed / segmentation 초기화
seeds = np.full(img.shape[:2], 0, np.int32)
segmentation = np.full(img.shape, 0, np.uint8)

# seed 개수
n_seeds = 9

# 색상 생성
colors = []
for i in range(n_seeds):
    colors.append((255 * i // n_seeds, randint(0, 255), randint(0, 255)))

# 상태 변수
mouse_pressed = False
current_seed = 1
seeds_updated = False


# =========================
# 마우스 콜백
# =========================
def mouse_callback(event, x, y, flags, param):
    global mouse_pressed, seeds_updated

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        cv2.circle(seeds, (x, y), 5, (current_seed), cv2.FILLED)
        cv2.circle(show_img, (x, y), 5, colors[current_seed - 1], cv2.FILLED)
        seeds_updated = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            cv2.circle(seeds, (x, y), 5, (current_seed), cv2.FILLED)
            cv2.circle(show_img, (x, y), 5, colors[current_seed - 1], cv2.FILLED)
            seeds_updated = True

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False


# =========================
# 윈도우 설정
# =========================
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

print("===== Watershed 사용법 =====")
print("1~9 : seed 번호 변경")
print("마우스 드래그 : seed 찍기")
print("c : 초기화")
print("ESC : 종료")


# =========================
# 메인 루프
# =========================
while True:
    cv2.imshow('segmentation', segmentation)
    cv2.imshow('image', show_img)

    k = cv2.waitKey(1)

    if k == 27:
        break

    elif k == ord('c'):
        show_img = np.copy(img)
        seeds = np.full(img.shape[:2], 0, np.int32)
        segmentation = np.full(img.shape, 0, np.uint8)

    elif k > 0 and chr(k).isdigit():
        n = int(chr(k))
        if 1 <= n <= n_seeds and not mouse_pressed:
            current_seed = n
            print(f"현재 seed: {current_seed}")

    # watershed 실행
    if seeds_updated and not mouse_pressed:
        seeds_copy = np.copy(seeds)
        cv2.watershed(img, seeds_copy)

        segmentation = np.full(img.shape, 0, np.uint8)

        for m in range(n_seeds):
            segmentation[seeds_copy == (m + 1)] = colors[m]

        seeds_updated = False

cv2.destroyAllWindows()