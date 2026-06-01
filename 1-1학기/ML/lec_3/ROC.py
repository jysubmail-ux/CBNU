import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

np.random.seed(42)

n = 300

# 🔹 실제값 (0/1 랜덤)
y_true = np.random.randint(0, 2, n)

# 🔹 score 생성 (겹치게 만들어야 곡선이 자연스러움)
y_score = np.where(
    y_true == 1,
    np.random.normal(0.65, 0.15, n),  # 양성
    np.random.normal(0.35, 0.15, n)   # 음성
)

# 🔹 0~1 범위 제한
y_score = np.clip(y_score, 0, 1)


fpr, tpr, thresholds = roc_curve(y_true, y_score)

print("===ROC 계산 결과===")
print(f"기준점(thresholds) : {thresholds}")
print(f"x축(FPR) : {fpr}")
print(f"y축(TPR) : {tpr}")
print()

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, marker='o', color='blue', linewidth=2, label='Our AI model')

plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Random (50%)')

plt.title('ROC Curve Analysis')
plt.xlabel('False Positive Rate(FPR)')
plt.ylabel('True Positive Rate(TPR)')
plt.grid(True)
plt.legend()

plt.show()