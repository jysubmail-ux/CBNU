import numpy as np
from sklearn.metrics import  accuracy_score

y_pred = np.array([0,1,1,0])
y_true = np.array([0,1,0,0])

manual_acc = sum(y_true == y_pred) / len(y_true)

print("1. 직접 계산한 정확도")
print(f"맞춘 갯수 (3) / 전체 갯수 (4) : {manual_acc}")
print()

sklearn_acc = accuracy_score(y_true, y_pred)

print("2. 사이킷런 함수 정확도")
print(f"결과 : {sklearn_acc}")
print()