import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles


X, y = make_circles(n_samples=200, factor=0.3, noise=0.05, random_state=42)

x1 = X[:, 0]
x2 = X[:, 1]


z = x1**2 + x2**2

fig = plt.figure(figsize=(16, 7))

ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(x1, x2, c=y, cmap='coolwarm', s=50, edgecolors='k')
ax1.set_title("Original 2D Space\n(Linearly Inseparable)", size=16)
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")


ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(x1, x2, z, c=y, cmap='coolwarm', s=50, edgecolors='k')

# 평면 생성
xx, yy = np.meshgrid(
    np.linspace(-1.2, 1.2, 10),
    np.linspace(-1.2, 1.2, 10)
)
zz = np.full(xx.shape, 0.4)

ax2.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

ax2.set_title("Projected 3D Space (z = x1^2 + x2^2)\n(Linearly Separable by a Plane)", size=16)
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.set_zlabel("z (New Dimension)")

ax2.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()