import os
import cv2
import joblib
import random
import numpy as np

TEST_DIR = "dataset_final/test"

# ======================
# 모델 로드
# ======================

svm = joblib.load("trash_svm.pkl")
encoder = joblib.load("label_encoder.pkl")


def imread_korean(path):
    return cv2.imdecode(
        np.fromfile(path, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )


# ======================
# HOG 설정
# ======================

hog = cv2.HOGDescriptor(
    _winSize=(64, 128),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)


# ======================
# 특징 추출
# ======================

def extract_feature(img):

    img = cv2.resize(img, (64, 128))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    hog_feature = hog.compute(gray).flatten()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])

    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()

    return np.hstack([
        hog_feature,
        hist_h,
        hist_s,
        hist_v
    ])


# ======================
# Test 이미지 수집
# ======================

samples = []

for cls in os.listdir(TEST_DIR):

    cls_path = os.path.join(TEST_DIR, cls)

    if not os.path.isdir(cls_path):
        continue

    for file in os.listdir(cls_path):

        path = os.path.join(cls_path, file)

        samples.append((path, cls))

print("전체 테스트 이미지:", len(samples))

random.shuffle(samples)

samples = samples[:10]

print("시연 이미지:", len(samples))


# ======================
# 시연
# ======================

for idx, (path, actual) in enumerate(samples, 1):

    img = imread_korean(path)

    if img is None:
        continue

    feature = extract_feature(img)

    pred = svm.predict([feature])[0]

    predict = encoder.inverse_transform([pred])[0]

    correct = actual == predict

    color = (0, 255, 0) if correct else (0, 0, 255)

    display = img.copy()

    cv2.putText(
        display,
        f"[{idx}/10]",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.putText(
        display,
        f"Actual : {actual}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.putText(
        display,
        f"Predict : {predict}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.imshow("Demo", display)

    print(
        f"[{idx}] Actual={actual} "
        f"Predict={predict} "
        f"{'O' if correct else 'X'}"
    )

    key = cv2.waitKey(0)

    if key == 27:
        break

cv2.destroyAllWindows()