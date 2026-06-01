import argparse
import cv2

parser  = argparse.ArgumentParser()
parser.add_argument('--path', default='Lena.png', help='Image path.')
params = parser.parse_args()

img = cv2.imread(params.path)
print('original image shape : ', img.shape)

width, height = 125, 256
resized_img = cv2.resize(img, (width, height))
print('resized image shape : ', resized_img.shape)

w_mul, h_mul = 0.25, 0.5
resized_img = cv2.resize(img, (0,0), resized_img, w_mul, h_mul)
print('image shape : ', resized_img.shape)

w_mul, h_mul = 2, 4
resized_img = cv2.resize(img, (0,0), resized_img, w_mul, h_mul, cv2.INTER_NEAREST)
print('image shape : ', resized_img.shape)


img_flip_along_x = cv2.flip(img, 0)
img_flip_along_x_along_y = cv2.flip(img_flip_along_x, 1)
img_flipped_xy = cv2.flip(img, -1)

assert img_flipped_xy.all() == img_flip_along_x_along_y.all()
