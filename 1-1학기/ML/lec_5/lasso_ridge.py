import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import make_regression

X,y = make_regression(n_samples=50, n_features=100, n_informative=10, noise=1, random_state=42)

lasso = Lasso(alpha=1.0).fit(X,y)
ridge = Ridge(alpha=1.0).fit(X,y)

print("="*60)
print("=== [가중치 값 비교 (상위 10개)] ===")
print("="*60)
df_comp = pd.DataFrame({
    'L2 (Ridge) 가중치 ' : ridge.coef_,
    'L1 (Lasso) 가중치 ' : lasso.coef_
})

print(df_comp.head(10))
print("\n" + "="*60)

print(f"L2 (Ridge)가 0으로 만든 변의 개수 : {np.sum(ridge.coef_ == 0)}개 / 100개")

print(f"L1 (Lasso)가 0으로 만든 변의 개수 : {np.sum(lasso.coef_ == 0)}개 / 100개")
print("="*60)

plt.figure(figsize=(15,6))

plt.subplot(1,2,1)
plt.stem(ridge.coef_, markerfmt=' ', basefmt="k-")
plt.title("L2 (Ridge) Coefficients \n(Many small non-zero values)", fontsize=14)
plt.xlabel("Feature Index (0~99)")
plt.ylabel("Coefficient Value")
plt.ylim(-100, 100)
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
plt.stem(lasso.coef_, markerfmt=' ', basefmt="k-")
plt.title("L1 (lasso) Coefficients \n(Sparse : Most are exactly 0)", fontsize=14)
plt.xlabel("Feature Index (0~99)")
plt.ylabel("Coefficient Value")
plt.ylim(-100, 100)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

