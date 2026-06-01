from hmac import digest_size

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=50, centers=2, random_state=6, cluster_std=1.5)
y[y == 0] = -1

svm_soft = SVC(kernel='linear', C=0.1)
svm_soft.fit(X,y)

svm_hard = SVC(kernel='linear', C=1000.0)
svm_hard.fit(X,y)

def print_alpha_info(model, name):
    alphas = np.abs(model.dual_coef_)[0]
    print(f"[{name} 결과]")
    print(f"설정된 C값: {model.C}")
    print(f"실제 알파(alpha)들의 최대값 : {np.max(alphas):.4f}")
    print(f"C 값에 도달한 서포트 백터 개수 : {np.sum(np.isclose(alphas, model.C))}개")
    print("-" * 30)


print_alpha_info(svm_soft, "Soft Margin")
print_alpha_info(svm_hard, "Hard Margin (High C)")
fig, axes = plt.subplots(1, 2, figsize=(14,6))
models = [svm_soft, svm_hard]
titles = ["Soft Margin (C = 0.1)", "Hard Margin (C = 1000.0)"]

for i, (model, title) in enumerate(zip(models, titles)):
    ax = axes[i]

    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s = 50, edgecolors='k')

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xx,yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contour(xx, yy, z, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--', '-', '--'])

    ax.scatter(model.support_vectors_[:,0], model.support_vectors_[:, 1],
               s=200, linewidth=2, facecolors='none', edgecolors='k')

    ax.set_title(f'Decision Boundary: {title}', size=16)

plt.tight_layout()
plt.show()

def print_equation(model, name):

    w = model.coef_[0]

    b = model.intercept_[0]

    print(f"[{name} 결정 경계 방정식]")
    print(f"w (가중치 벡터): [{w[0]:.4f}, {w[1]:.4f}]")
    print(f"b (편향/절편): {b:.4f}")

    print(f"방정식: {w[0]:.4f}*x1 + {w[1]:.4f}*x2 + {b:.4f} = 0")
    print("-" * 40)


print_equation(svm_soft, "Soft Margin (C=0.1)")
print_equation(svm_hard, "Hard Margin (C=1000.0)")