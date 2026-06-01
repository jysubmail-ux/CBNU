import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. 스마트 팩토리 센서 데이터 시뮬레이션
# 총 36개의 센서(변수) 중 5개만 실제 불량과 직결된 핵심 센서로 설정

X, y = make_classification(
    n_samples=1500,
    n_features=36,
    n_informative=5,
    n_redundant=2,
    random_state=42
)


# 훈련 데이터와 테스트 데이터 분리 (8:2)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 2. 랜덤 포레스트 모델 구축 및 하이퍼파라미터 세팅 (핵심)

rf_model = RandomForestClassifier(
    n_estimators=500,      # 500개의 트리 생성
    max_features=None,   # 랜덤 서브스페이스
    oob_score=True,        # OOB 검증
    random_state=42,
    n_jobs=-1              # CPU 전체 사용
)


# 3. 모델 학습
rf_model.fit(X_train, y_train)


# 4. 검증 및 평가

# OOB Score 출력
print(f"OOB Score (자체 검증 점수): {rf_model.oob_score_:.4f}")

# 테스트 데이터 예측
y_pred = rf_model.predict(X_test)

# 테스트 정확도
print(
    f"Test Accuracy (실제 테스트 점수): "
    f"{accuracy_score(y_test, y_pred):.4f}"
)

print("\n[분류 리포트]")
print(classification_report(y_test, y_pred))


# 5. Feature Importance (변수 중요도) 시각화

# 각 센서 중요도
importances = rf_model.feature_importances_

# 중요도 내림차순 정렬
indices = np.argsort(importances)[::-1]


# 상위 10개 핵심 센서 시각화
plt.figure(figsize=(10, 6))

plt.title(
    "Smart Factory Sensor Feature Importances (Top 10)"
)

plt.bar(
    range(10),
    importances[indices][:10],
    align="center",
    color="steelblue"
)

plt.xticks(
    range(10),
    [f"Sensor_{i}" for i in indices[:10]],
    rotation=45
)

plt.xlim(-1, 10)

plt.ylabel("Importance Score")

plt.tight_layout()
plt.show()