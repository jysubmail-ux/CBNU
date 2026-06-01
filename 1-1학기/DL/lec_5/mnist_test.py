from tensorflow.keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

y_train = y_train.reshape(-1, 784)
y_test = y_test.reshape(-1, 784)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)