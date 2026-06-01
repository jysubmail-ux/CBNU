import numpy as np
import cv2

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

data = np.load('./stereo/case1/stereo.npy').item()

Kl = data['Kl']
Dl = data['Dl']
Kr = data['Kr']
Dr = data['Dr']
left_pts = data['left_pts']
right_pts = data['right_pts']
E_from_streo = data['E']
F_from_streo = data['F']

left_pts = np.vstack(left_pts)
right_pts = np.vstack(right_pts)

left_pts = cv2.undistortPoints(left_pts, Kl, Dl, P=Kl)
right_pts = cv2.undistortPoints(right_pts, Kr, Dr, P=Kr)

F, mask = cv2.findFundamentalMat(left_pts, right_pts, cv2.FM_LMEDS)

E = Kr.T @ F @ Kl

print('Fundamental Matrix:')
print(F)
print('Essential matrix:')
print(E)