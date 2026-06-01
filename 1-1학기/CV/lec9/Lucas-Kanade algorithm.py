import cv2
import numpy as np

'''
traffic.mp4 읽기
goodFeaturesToTrack() 로 초기 특징점 추출
calcOpticalFlowPyrLK() 로 프레임간 이동 추적
tracks에 누적
녹색 점 표시
ESC: 종료
c: tracking 초기화

calcOpticalFlowPyrLK 이거는 dense 하진 않다
첫번째 이미지에서 initial 점을 구한 뒤, 다음 frame 에서 그 점을 찾는 알고리즘
'''

video = cv2.VideoCapture('traffic.mp4')
prev_pts = None
prev_gray_frame = None
tracks = None

while True:

    retval, frame = video.read()

    if not retval:
        break

    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if prev_pts is not None:
        pts, status, errors = cv2.calcOpticalFlowPyrLK(prev_gray_frame, gray_frame,prev_pts,None,
            winSize=(15, 15),
            maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        good_pts = pts[status == 1]

        if tracks is None:
            tracks = good_pts

        else:
            tracks = np.vstack((tracks, good_pts))
        for p in tracks:
            cv2.circle(frame,(int(p[0]), int(p[1])),3,(0, 255, 0), -1)

    else:

        pts = cv2.goodFeaturesToTrack(gray_frame,500,0.05,10)

        pts = pts.reshape(-1, 1, 2)

    prev_pts = pts
    prev_gray_frame = gray_frame

    cv2.imshow('frame', frame)

    key = cv2.waitKey() & 0xff

    if key == 27:
        break

    if key == ord('c'):

        tracks = None
        prev_pts = None


cv2.destroyAllWindows()