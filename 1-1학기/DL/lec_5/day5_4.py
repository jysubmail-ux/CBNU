import cv2 as cv
import numpy as np

img = cv.imread('soccer.jpg')
img = cv.resize(img, dsize=(0,0), fx=0.4, fy=0.4) # 크기를 40% 축소
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.putText(gray, 'soccer', (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2) # 글씨 'soccer'를 (10,20) 위치에 흰색(255,255,255) 로 입력
cv.imshow('Original', gray)
cv.waitKey()

smooth = np.hstack((cv.GaussianBlur(gray, (5,5), 0.0), cv.GaussianBlur(gray, (9,9), 0.0), cv.GaussianBlur(gray, (15,15), 0.0)))
'''
1. hstack : 그림을 가로로 붙이는 역할
2. 각각의 그림은 가우시안 필터 적용
3. 오른쪽으로 갈수록 저점 외곽추출 힘듬
'''
cv.imshow('Smooth', smooth)
cv.waitKey()

# 엠보싱 필터 : 대각선 방향 밝기 차이 강조
femboss = np.array([[-1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0]])

gray16 = np.int16(gray) # 필터 연산시 음수 발생 가능성. int8의 경우 음수 표현을 못함

#2D필터 적용 밝기 128, clip을 통해 0~255 제한, 이후 uint8로 이미지 타입 변환
emboss = np.uint8(np.clip(cv.filter2D(gray16, -1, femboss)+128, 0, 255))
emboss_bad = np.uint8(cv.filter2D(gray16, -1, femboss)+128)
emboss_worse = cv.filter2D(gray,-1,femboss)

cv.imshow('Emboss', emboss)
cv.imshow('Emboss_bad', emboss_bad)
cv.imshow('Emboss_worse', emboss_worse)

cv.waitKey()
cv.destroyAllWindows()