import cv2
import numpy as np
import os


'''
1. 이미지를 읽고
2. (10, 7) 사이즈 findChessboardCorners를 찾음
3. 찾은걸 화면에 그려줌
4. s 버튼의 의미 : 구해진 코너값을 샘플에 넣은 뒤 코너점을 변환
5. 샘플의 갯수만큼 for문 동작, 찾은 코너를 계산

'''

pattern_size = (10, 7)
samples = []
file_list = os.listdir('./pinhole_calib')
img_file_list = [file for file in file_list if file.startswith('img')]

for filename in img_file_list:

    frame = cv2.imread(os.path.join('./pinhole_calib', filename))
    res, corners = cv2.findChessboardCorners(frame, pattern_size)
    img_show = np.copy(frame)

    cv2.drawChessboardCorners(img_show, pattern_size, corners, res)
    cv2.putText(img_show,'Samples captured: %d' % len(samples),(0, 40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0, 255, 0),2)
    cv2.imshow('chessboard', img_show)
    wait_time = 0 if res else 30

    k = cv2.waitKey(wait_time)

    if k == ord('s') and res:
        samples.append(
            (
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                corners
            )
        )

    elif k == 27:
        break

cv2.destroyAllWindows()

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


for i in range(len(samples)):

    img, corners = samples[i]
    corners = cv2.cornerSubPix(img,corners, (10, 10), (-1, -1), criteria)


pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
images, corners = zip(*samples)
pattern_points = [pattern_points] * len(corners)


rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
    pattern_points,
    corners,
    images[0].shape,
    None,
    None
)


np.save('camera_mat.npy', camera_matrix)
np.save('dist_coefs.npy', dist_coefs)

print(np.load('camera_mat.npy'))
print(np.load('dist_coefs.npy'))