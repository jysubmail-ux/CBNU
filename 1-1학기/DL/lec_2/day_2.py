import numpy as np
import matplotlib.pyplot as plt

# x 값 생성 (0 ~ 2π)
x = np.linspace(0, 2*np.pi, 100)

# sin 계산
y = np.cos(x)

# 그래프 그리기
plt.plot(x, y)

plt.title("Sin Graph")
plt.xlabel("x")
plt.ylabel("sin(x)")

plt.show()