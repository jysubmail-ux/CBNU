import numpy as np


# 🔹 sigmoid 함수
# 선형 결과(z)를 0~1 사이 확률로 변환
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 🔹 가설 함수 (Hypothesis function)
# 입력값 x와 가중치 theta를 이용해 예측 확률 계산
def hypothesis_func(x, theta):
    # 🔸 선형 결합 (z = x·theta)
    # (행렬 X) · (가중치 theta) → 각 샘플별 점수 z 생성
    z = np.dot(x, theta)

    # 🔸 sigmoid 적용 → 확률로 변환
    return sigmoid(z)


# 🔹 비용 함수 (Cost Function)
# 예측값(h)과 실제값(y)의 차이를 수치로 계산 (Cross Entropy)
def compute_cost(x, y, theta):
    # 🔸 데이터 개수
    m = y.shape[0]

    # 🔸 모델의 예측 확률
    h = hypothesis_func(x, theta)

    # 🔸 정답이 1일 때의 비용
    # y=1이면 log(h)가 커야 좋음 (h가 1에 가까울수록)
    term1 = y.T.dot(np.log(h))

    # 🔸 정답이 0일 때의 비용
    # y=0이면 log(1-h)가 커야 좋음 (h가 0에 가까울수록)
    term2 = (1 - y).T.dot(np.log(1 - h))

    # 🔸 최종 비용 (평균 + 음수)
    # → 값이 작을수록 좋은 모델
    J = (-1.0 / m) * (term1 + term2)

    return J


# 🔹 입력 데이터 (특징 2개짜리 샘플 3개)
# 예: [온도, 압력]
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

# 🔹 정답 데이터 (라벨)
# 1 = 고장, 0 = 정상
y = np.array([1, 0, 1])

# 🔹 가중치 (모델 파라미터)
theta = np.array([0.001, 0.002])

# 🔹 1. 예측 확률 계산
prediction = hypothesis_func(X, theta)

print("=== 1. 모델의 예측 확률===")
print(prediction)
print()

# 🔹 2. 비용 계산
cost = compute_cost(X, y, theta)

print("=== 2. 현재 모델의 비용(Cost) ===")

#  출력 포맷 수정 필요 (.4f 위치 오류였음)
print(f"점수 : {cost:.4f}")