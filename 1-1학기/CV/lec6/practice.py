import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('Lena.png')


data_rgb = image.reshape((-1, 3)).astype(np.float32)


h, w = image.shape[:2]
xs, ys = np.meshgrid(np.arange(w), np.arange(h))
xs = xs.reshape((-1,1)).astype(np.float32) / w
ys = ys.reshape((-1,1)).astype(np.float32) / h

data_rgbxy = np.hstack([data_rgb/255.0, xs, ys])

num_class = 8
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)


_, labels_rgb, centres_rgb = cv2.kmeans(
    data_rgb, num_class, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

centres_rgb = np.uint8(centres_rgb)
seg_rgb = centres_rgb[labels_rgb.flatten()].reshape(image.shape)


_, labels_rgbxy, centres_rgbxy = cv2.kmeans(
    data_rgbxy, num_class, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

seg_rgbxy = np.zeros_like(image.reshape((-1,3)))

for i in range(num_class):
    seg_rgbxy[labels_rgbxy.flatten()==i] = np.mean(
        image.reshape((-1,3))[labels_rgbxy.flatten()==i], axis=0
    )

seg_rgbxy = seg_rgbxy.reshape(image.shape).astype(np.uint8)


plt.figure(figsize=(12,6))

plt.subplot(1,3,1)
plt.title('original')
plt.axis('off')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1,3,2)
plt.title('RGB')
plt.axis('off')
plt.imshow(cv2.cvtColor(seg_rgb, cv2.COLOR_BGR2RGB))

plt.subplot(1,3,3)
plt.title('RGBXY')
plt.axis('off')
plt.imshow(cv2.cvtColor(seg_rgbxy, cv2.COLOR_BGR2RGB))

plt.show()



img = cv2.imread('Lena.png', cv2.IMREAD_COLOR)
show_img = np.copy(img)

mouse_pressed = False
x=y=w=h=0


def mouse_callback(event, _x, _y, flags, param):
    global show_img, x,y,w,h,mouse_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        x, y = _x, _y
        show_img = np.copy(img)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            show_img = np.copy(img)
            cv2.rectangle(show_img, (x,y), (_x, _y), (0,255,0),3)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        w, h = _x-x , _y-y


cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

print(" 사각형 드래그 후 'a' 누르세요")

while True:
    cv2.imshow('image', show_img)
    k = cv2.waitKey(1)

    if k == ord('a') and not mouse_pressed:
        if w * h > 0:
            break

cv2.destroyAllWindows()


labels = np.zeros(img.shape[:2], np.uint8)
labels, bgdModel, fgdModel = cv2.grabCut(
    img, labels, (x,y,w,h), None, None, 5, cv2.GC_INIT_WITH_RECT
)

show_img = np.copy(img)
show_img[(labels == cv2.GC_PR_BGD) | (labels == cv2.GC_BGD)] //= 3

cv2.imshow('image', show_img)
cv2.waitKey()
cv2.destroyAllWindows()


label = cv2.GC_BGD
lbl_clrs = {cv2.GC_BGD:(0,0,0), cv2.GC_FGD:(255,255,255)}

def mouse_callback(event, x, y, flags, param):
    global mouse_pressed, label

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        cv2.circle(labels, (x,y), 5, label, cv2.FILLED)
        cv2.circle(show_img, (x,y), 5, lbl_clrs[label], cv2.FILLED)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            cv2.circle(labels, (x,y), 5, label, cv2.FILLED)
            cv2.circle(show_img, (x,y), 5, lbl_clrs[label], cv2.FILLED)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False


cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

print("0: 배경 / 1: 전경 / a: 실행")

while True:
    cv2.imshow('image', show_img)
    k = cv2.waitKey(1)

    if k == ord('a') and not mouse_pressed:
        break
    elif k == ord('0'):
        label = cv2.GC_BGD
    elif k == ord('1'):
        label = cv2.GC_FGD

cv2.destroyAllWindows()


labels, bgdModel, fgdModel = cv2.grabCut(
    img, labels, None,
    bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK
)

show_img = np.copy(img)
show_img[(labels == cv2.GC_PR_BGD) | (labels == cv2.GC_BGD)] //= 3

cv2.imshow('result', show_img)
cv2.waitKey()
cv2.destroyAllWindows()