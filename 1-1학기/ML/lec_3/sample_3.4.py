import numpy as np

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

def minimize_gradient(x,y,theta, iterations=1000, alpha=0.01):
    m = y.size
    cost_history = []

    for _ in range(iterations):
        h = hypothesis_func(x, theta)
        loss = h-y
        gradiant = x.T.dot(loss)/m

        theta = theta -(alpha* gradiant)

        if(_ % 100) == 0:
            current_cost = compute_cost(x,y,theta)
            cost_history.append(current_cost)
            print(f"반복 횟수 : {_:>4} : 현재 비용 = {current_cost:.5f}")
    return theta, cost_history

x_test = np.array([[1,2], [3,4], [5,6]])
y_test = np.array([1,0,1])
initial_theta = np.array([0.0, 0.0])

print("학습을 시작합니다...")

final_theta, history = minimize_gradient(x_test, y_test, initial_theta, iterations=1000, alpha=0.01)

print("학습 완료")
print(f"최종 가중치 : {final_theta}")
print(f"최종 비용 : {compute_cost(x_test, y_test, final_theta):.5f}")
