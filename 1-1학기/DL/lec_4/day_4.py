import numpy as np
import matplotlib.pyplot as plt

# x 범위
x = np.linspace(-10, 10, 100)

# Sigmoid
sigmoid = 1 / (1 + np.exp(-x))

# Step function (계단함수)
step = np.where(x >= 0, 1, 0)

relu = np.maximum(0, x)

# 그래프
plt.figure(figsize=(15,4))

plt.subplot(1, 3, 1)
plt.plot(x, sigmoid)
plt.title("Sigmoid")
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(x, step)
plt.title("Step")
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(x, relu)
plt.title("Relu")
plt.grid()