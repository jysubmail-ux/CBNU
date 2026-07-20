import cv2
import os
import joblib
import numpy as np

from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


TRAIN_DIR = "dataset_final/train"
TEST_DIR = "dataset_final/test"


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

    # 비율 유지 resize
    img = cv2.resize(img, (64, 128))

    # ----------------------
    # Histogram Equalization
    # ----------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # ----------------------
    # HOG 특징
    # ----------------------
    hog_feature = hog.compute(gray).flatten()

    # ----------------------
    # HSV Color Histogram
    # ----------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])

    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()

    color_feature = np.hstack([
        hist_h,
        hist_s,
        hist_v
    ])

    # ----------------------
    # 최종 특징 벡터
    # ----------------------
    feature = np.hstack([
        hog_feature,
        color_feature
    ])

    return feature


# ======================
# Train 로딩
# ======================

X_train = []
y_train = []

print("Train 로딩...")

for cls in os.listdir(TRAIN_DIR):

    cls_path = os.path.join(TRAIN_DIR, cls)

    if not os.path.isdir(cls_path):
        continue

    count = 0

    for file in os.listdir(cls_path):

        path = os.path.join(cls_path, file)

        img = imread_korean(path)

        if img is None:
            continue

        feature = extract_feature(img)

        X_train.append(feature)

        y_train.append(cls)

        count += 1

    print(cls, count)

print("\nTrain:", len(X_train))
print(Counter(y_train))


# ======================
# Test 로딩
# ======================

X_test = []
y_test = []

print("\nTest 로딩...")

for cls in os.listdir(TEST_DIR):

    cls_path = os.path.join(TEST_DIR, cls)

    if not os.path.isdir(cls_path):
        continue

    count = 0

    for file in os.listdir(cls_path):

        path = os.path.join(cls_path, file)

        img = imread_korean(path)

        if img is None:
            continue

        feature = extract_feature(img)

        X_test.append(feature)

        y_test.append(cls)

        count += 1

    print(cls, count)

print("\nTest:", len(X_test))
print(Counter(y_test))


# ======================
# Label Encoding
# ======================

encoder = LabelEncoder()

y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)

print("\n클래스:", encoder.classes_)


# ======================
# RBF SVM
# ======================

print("\nSVM 학습 중...")
w
svm = make_pipeline(
    StandardScaler(),
    SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        random_state=42
    )
)

svm.fit(X_train, y_train_enc)


# ======================
# 평가
# ======================

pred = svm.predict(X_test)

acc = accuracy_score(
    y_test_enc,
    pred
)

print("\nAccuracy:", acc)

print("\nClassification Report")

print(
    classification_report(
        y_test_enc,
        pred,
        target_names=encoder.classes_
    )
)

# ======================
# Confusion Matrix
# ======================

cm = confusion_matrix(y_test_enc, pred)

plt.figure(figsize=(8, 6))

plt.imshow(cm, cmap='Blues')

plt.title("Confusion Matrix")

plt.colorbar()

tick_marks = np.arange(len(encoder.classes_))

plt.xticks(
    tick_marks,
    encoder.classes_,
    rotation=45
)

plt.yticks(
    tick_marks,
    encoder.classes_
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="black"
        )

plt.ylabel("Actual")
plt.xlabel("Predicted")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nconfusion_matrix.png 저장 완료")

# ======================
# 저장
# ======================

joblib.dump(
    svm,
    "trash_svm.pkl"
)

joblib.dump(
    encoder,
    "label_encoder.pkl"
)

print("\n모델 저장 완료")