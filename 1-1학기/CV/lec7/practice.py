import cv2
import numpy as np

# =========================
# 이미지 로드
# =========================
img = cv2.imread('scenetext01.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =========================
# 1. Harris Corner
# =========================
harris = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
harris = cv2.dilate(harris, None)

harris_img = img.copy()
harris_img[harris > 0.01 * harris.max()] = (0, 0, 255)  # 🔴 빨강

# =========================
# 2. FAST
# =========================
fast = cv2.FastFeatureDetector_create(30, True)
kp_fast = fast.detect(gray)

fast_img = img.copy()
for p in cv2.KeyPoint_convert(kp_fast):
    cv2.circle(fast_img, tuple(p), 3, (0, 255, 0), -1)  # 🟢 초록 원

# =========================
# 3. Good Features To Track
# =========================
gftt = cv2.goodFeaturesToTrack(gray, 150, 0.01, 10)

gftt_img = img.copy()
if gftt is not None:
    for c in gftt:
        x, y = c.ravel()
        # 🔵 파랑 사각형
        cv2.rectangle(gftt_img, (int(x)-3, int(y)-3), (int(x)+3, int(y)+3), (255, 0, 0), 1)

# =========================
# 4. SIFT
# =========================
sift = cv2.xfeatures2d.SIFT_create(150)
kp_sift, _ = sift.detectAndCompute(gray, None)

sift_img = img.copy()
for kp in kp_sift:
    x, y = int(kp.pt[0]), int(kp.pt[1])
    # 🟡 노랑 원 + 크기 반영
    cv2.circle(sift_img, (x, y), int(kp.size/10), (0, 255, 255), 1)

# =========================
# 이미지 크기 맞추기
# =========================
h, w = img.shape[:2]

harris_img = cv2.resize(harris_img, (w, h))
fast_img   = cv2.resize(fast_img, (w, h))
gftt_img   = cv2.resize(gftt_img, (w, h))
sift_img   = cv2.resize(sift_img, (w, h))

# =========================
# 제목 넣기 함수
# =========================
def put_title(image, title):
    out = image.copy()
    cv2.putText(out, title, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)
    return out

harris_img = put_title(harris_img, "Harris")
fast_img   = put_title(fast_img, "FAST")
gftt_img   = put_title(gftt_img, "GFTT")
sift_img   = put_title(sift_img, "SIFT")

# =========================
# 한 화면에 2x2 배치
# =========================
top = np.hstack((harris_img, fast_img))
bottom = np.hstack((gftt_img, sift_img))

final = np.vstack((top, bottom))

# =========================
# 출력
# =========================

scale = 0.5  # 50% 축소
final_small = cv2.resize(final, None, fx=scale, fy=scale)

cv2.imshow('Feature Comparison', final_small)

cv2.waitKey(0)
cv2.destroyAllWindows()
