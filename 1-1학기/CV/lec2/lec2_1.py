import argparse

import cv2

parser  = argparse.ArgumentParser()
parser.add_argument('--path', default='Lena.png', help='Image path.')
params = parser.parse_args()

img = cv2.imread(params.path)

print('read {}'.format(params.path))
print('shape : ', img.shape)
print('dtype : ', img.dtype)

img = cv2.imread(params.path, cv2.IMREAD_GRAYSCALE)

print('read {} as grayscale'.format(params.path))
print('shape : ', img.shape)
print('dtype : ', img.dtype)