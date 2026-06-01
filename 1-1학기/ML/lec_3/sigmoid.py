import numpy as np
import matplotlib.pyplot as plt

# 🔹 sigmoid 함수 정의
# 입력값 z를 확률값(0~1)으로 변환하는 함수
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


print("[part 1] sigmoid함수 수치 테스트")

# 🔹 z = 0일 때 → 0.5 (중앙값)
s_0 = sigmoid(0)
print(f"입력값이 0일때 : ", s_0)

# 🔹 z = 100일 때 → 거의 1
s_100 = sigmoid(30)
print(f"입력값이 100일때 : ", s_100)

# 🔹 z = -100일 때 → 거의 0
s_m100 = sigmoid(-10)
print(f"입력값이 -100일때 : ", s_m100)


# 🔹 -10 ~ 10 구간 생성 (그래프용)
z_value = np.linspace(-10, 10, 200)

# 🔹 sigmoid 적용 → 확률값 생성
probabilities = sigmoid(z_value)


# 🔹 그래프 생성
plt.figure(figsize=(10,6))

# sigmoid 곡선
plt.plot(z_value, probabilities, color='red', linewidth=3, label="Sigmoid Curve")

# y=1, y=0 기준선
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)

# 🔥 분류 기준선 (Threshold = 0.5)
plt.axhline(y=0.5, color='black', linestyle=':', label='Threshold (0.5)')

# x=0 기준선
plt.axvline(x=0.0, color='gray', linestyle='--', alpha=0.5)

# sigmoid(0) = 0.5 포인트 표시
plt.scatter(0, 0.5, color='blue', s=100, zorder=5, label='sigmoid(0)')

# 제목 및 라벨
plt.title('Sigmoid Function (Magic Compressor)')
plt.xlabel('Raw Score (z) - from AI model')
plt.ylabel('Probability (0.0 ~ 1.0) - output')

# 그리드 + 범례
plt.grid(True, linestyle='-', alpha=0.3)
plt.legend(loc='upper left')

plt.show()


# ❌ 잘못 입력된 코드 (오타로 인한 SyntaxError 발생)
# wntjr sjgdjwnj