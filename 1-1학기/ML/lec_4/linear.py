import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
X = 200* np.random.rand(100,1) # X 값을 0~100 에서 랜덤값 200개
y = 30 * X + 10 + np.random.randn(100,1) * 10 # 정답값

model = LinearRegression()
model.fit(X,y)

w1 = model.coef_[0][0] # 기울기 , 2차원 배열이라 [0][0] 2차원 배열
w0 = model.intercept_[0] # 절편

print(f"=== Training Result ===")
print(f"Estimated Slope (w1) : {w1:.2f}")
print(f"Estimated Intercept (w0): {w0:.2f}")
print(f"Final Equation: y = {w1:.2f}x + {w0:.2f}")

y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\n=== Model Evaluation ===")
print(f"Mean Squared Error (MSE) : {mse:.2f}")
print(f"R-squared (R2 Score) : {r2:.2f}")

plt.figure(figsize=(10,6))

plt.scatter(X, y, color='blue', alpha=0.6, label='Actual Data')

plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')

plt.title("Linear Regression: Watcha Likes vs Audience", fontsize=14)
plt.xlabel("Watcha 'Like' Count", fontsize=12)
plt.ylabel("Total Audience Count", fontsize=12)
# plt.ylim(0, 100)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()
