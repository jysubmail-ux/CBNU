'''
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('jyp.jpg')

plt.imshow(img)
plt.show()


import cv2

# 이미지 읽기
image = cv2.imread(img_path)

# 이미지가 정상적으로 읽혔는지 확인
if image is None:
    print("이미지를 불러오지 못했습니다. 경로를 확인하세요.")
else:
    # 이미지 출력
    cv2.imshow("Image", image)

    # 키 입력 대기 (0 = 무한 대기)
    cv2.waitKey(0)

    # 창 닫기
    cv2.destroyAllWindows()
	


def AND(x1, x2):
	w1, w2, theta = 0.5, 0.5, 0.7
	tmp = x1*w1 + x2*w2
	if tmp <= theta:
		return 0
	elif tmp > theta:
		return 1

print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))
'''
import numpy as np


def AND(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])
	b = -0.7
	
	tmp = np.sum(w*x) + b
	if tmp <= 0:
		return 0
	else:
		return 1
	
def NAND(x1, x2):
	x = np.array([x1, x2])
	w = np.array([-0.5, -0.5])
	b = 0.7
	
	tmp = np.sum(w*x) + b
	if tmp <= 0:
		return 0
	else:
		return 1
	
	
def OR(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])
	b = -0.2
	
	tmp = np.sum(w*x) + b
	if tmp <= 0:
		return 0
	else:
		return 1

def XOR(x1, x2):
	s1 = NAND(x1, x2)
	s2 = OR(x1,x2)
	y = AND(s1, s2)
	return y

print("==========AND===========")
print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))
print("==========NAND===========")
print(NAND(0,0))
print(NAND(0,1))
print(NAND(1,0))
print(NAND(1,1))
print("==========OR===========")
print(OR(0,0))
print(OR(0,1))
print(OR(1,0))
print(OR(1,1))
print("==========XOR===========")
print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))
