import cv2
import numpy as np
import matplotlib.pyplot as plt

img0 = cv2.imread('people.jpg', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('face.jpeg', cv2.IMREAD_GRAYSCALE)

detector = cv2.ORB_create(500)

# fea0, fea1 은 각각의 특징벡터
_, fea0 = detector.detectAndCompute(img0, None)
_, fea1 = detector.detectAndCompute(img1, None)

descr_type = fea0.dtype

# 각각의 특징을 KMeans로 50개의 그룹으로 묶음
bow_trainer = cv2.BOWKMeansTrainer(50)
bow_trainer.add(np.float32(fea0))
bow_trainer.add(np.float32(fea1))
vocab = bow_trainer.cluster().astype(descr_type)


bow_descr = cv2.BOWImgDescriptorExtractor(detector, cv2.BFMatcher(cv2.NORM_HAMMING))
bow_descr.setVocabulary(vocab)

# Lena를 읽은 뒤, 특징점을 찾고 각각의 vocabulary 매칭
img = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
kps = detector.detect(img, None)

descr = bow_descr.compute(img, kps)

# 이미지 시각화
plt.figure(figsize=(10, 3))
plt.title('image BoW descriptor')
plt.bar(np.arange(len(descr[0])), descr[0])
plt.xlabel('vocabulary element')
plt.ylabel('frequency')
plt.tight_layout()
plt.show()