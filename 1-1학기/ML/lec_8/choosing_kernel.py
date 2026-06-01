import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_circles(n_samples=300, factor = 0.3, noise=0.15, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

models = {
    "1. Linear Kernel ": SVC(kernel='linear', C=1.0),
    "2. RBF Kernel ": SVC(kernel='rbf', C=1.0, gamma=0.5),
    "3. RBF Kernel ": SVC(kernel='rbf', C=1.0, gamma=50.0)
}

ESTIMATED_BAYES_ERROR = 0.05

print("=" * 50)
print(f"[이론적 한계] Bayes Error (Irreducible Error): 약 {ESTIMATED_BAYES_ERROR * 100:.1f}% \n"
      f" (데이터 생성 시 부여된 15%의 노이즈로 인해 발생하는 본질적 겹침)")
print("=" * 50)

fig, axes = plt.subplots(1,3, figsize=(18,5))
for i, (title, model) in enumerate(models.items()):

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_error = 1.0 - accuracy_score(y_train, y_train_pred)
    test_error = 1.0 - accuracy_score(y_test, y_test_pred)

    print(f"[{title}]")
    print(f" - Train Error: {train_error * 100:.1f}%")
    print(f" - Test Error : {test_error * 100:.1f}%")
    print("-" * 50)

    ax = axes[i]

    ax.scatter(X_train[:,0], X_train[:, 1], c=y_train, cmap='coolwarm', s=20, alpha=0.3)
    ax.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='coolwarm', s=60, edgecolors='k', marker='*')

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contour(xx,yy,z,colors='k', levels=[0], alpha=0.8, linestyles=['-'])
    ax.set_title(f"{title}\nTest Err:{test_error * 100:.1f}%", size=13)

plt.tight_layout()
plt.show()