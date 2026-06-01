import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

mnist = keras.datasets.mnist


(x_train, t_train), (x_test, t_test) = mnist.load_data()

print(len(x_train))  # x_train 배열의 크기를 출력

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# flatten
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

img = x_train[99]
label = t_train[99]
print(label)

img = img.reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.title(f"label: {label}")
plt.axis('off')
plt.show()