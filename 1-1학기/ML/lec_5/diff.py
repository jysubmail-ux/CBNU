import numpy as np
import matplotlib.pyplot as plt

# 샘플 데이터
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# 비용함수 (MSE)
def compute_cost(w):
    y_pred = w * X
    return np.mean((y - y_pred) ** 2)

# w 범위
w_values = np.linspace(0, 2, 100)
cost_values = [compute_cost(w) for w in w_values]

# ===== 경사하강법 =====
w = 0  # 초기값
alpha = 0.1
history_w = []
history_cost = []

for _ in range(10):
    y_pred = w * X
    gradient = -2 * np.mean(X * (y - y_pred))
    w = w - alpha * gradient

    history_w.append(w)
    history_cost.append(compute_cost(w))

# ===== 최소자승법 (정답) =====
w_opt = np.sum(X * y) / np.sum(X ** 2)
cost_opt = compute_cost(w_opt)

# ===== 그래프 =====
plt.figure(figsize=(10, 6))

# 비용함수 곡선
plt.plot(w_values, cost_values, label="Cost Function", linewidth=2)

# 경사하강법 경로
plt.scatter(history_w, history_cost, color='red', label="Gradient Descent Path")

# 최소자승법 해
plt.scatter(w_opt, cost_opt, color='blue', s=100, label="OLS Solution")

plt.title("OLS vs Gradient Descent")
plt.xlabel("Weight (w)")
plt.ylabel("Cost (MSE)")
plt.legend()
plt.grid()
plt.show()