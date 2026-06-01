import numpy as np


# 🔹 sigmoid 함수
# 선형 결과(z)를 0~1 사이 확률로 변환
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 🔹 가설 함수 (Hypothesis function)
# 입력값 x와 가중치 theta를 이용해 예측 확률 계산
def hypothesis_func(x, theta):
    # 🔸 선형 결합 (z = x·theta)
    # 각 입력값에 가중치를 곱해서 더함
    z = np.dot(x, theta)

    # 🔸 sigmoid 적용 → 확률로 변환
    return sigmoid(z)


# 🔹 입력 데이터 (특징값)
# 예: [온도, 진동값] 같은 센서 데이터라고 보면 됨
x_data = np.array([30, 0.8])

# 🔹 가중치 (모델이 학습한 값)
# 각 특징이 결과에 얼마나 영향을 주는지
theta_data = np.array([0.1, 5.0])

# 🔹 최종 확률 계산
prob = hypothesis_func(x_data, theta_data)

# 🔹 선형 점수 z 출력
# z = (30 * 0.1) + (0.8 * 5.0)
print(f"종합점수(z) : {np.dot(x_data, theta_data)}")

# 🔹 sigmoid를 통과한 최종 확률 출력
# 고장일 확률 (%)로 표현
print(f"최종 고장 확률 : {prob * 100:.2f}%")