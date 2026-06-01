import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

'''
X : 랜덤 값
y + sin(x)

기존 y 값에 노이즈 추가

regr_1 , regr_2 는 각각의 의사결정 트리 depth만 다른 경우
그래프 확인 결과 depth = 5인 경우가 조금 더 정확하게 표현
'''


np.random.seed(42)

X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()


y[::5] += 3 * (0.5 - np.random.rand(16))


regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)

regr_1.fit(X, y)
regr_2.fit(X, y)


X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)


plt.figure(figsize=(10, 6))

plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)

plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression (Step-function Prediction)")
plt.legend()
plt.show()

plt.figure(figsize=(20, 8))

plot_tree(regr_1, filled=True, feature_names=["X"])
plt.title("Regression Tree Structure (MSE Minimization)")

plt.show()