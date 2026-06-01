"""
행렬 계산 관련
딥러닝 관련 모든 데이터는 다차원배열을 통해 저장, 활용한다
numpy를 통해 계산
"""
import numpy as np
'''
import numpy as np
A = np.array([[1,2], [3,4], [5,6]])
print(A.shape)
print(A.ndim)
B = np.array([7,8])
print(B.shape)
print(B.ndim)
Y = np.dot(A,B)
print(Y)
print(Y.shape)
print(Y.ndim)
'''

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
print(A1)