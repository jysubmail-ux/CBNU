import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Lena.png')
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

data = image_lab.reshape((-1, 3)).astype(np.float32)

num_class = 8
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
_, labels, centres = cv2.kmeans(data, num_class, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centres = np.uint8(centres)
segmented_lab = centres[labels.flatten()]
segmented_lab = segmented_lab.reshape(image.shape)

segmented = cv2.cvtColor(segmented_lab, cv2.COLOR_Lab2BGR)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title('original')
plt.axis('off')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1,2,2)
plt.title('segmented')
plt.axis('off')
plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))

plt.show()