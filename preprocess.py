import pprint

import numpy as np
import matplotlib.pyplot as plt
import cv2


def preprocess():
    image = cv2.imread(f'data/2.jpg', cv2.IMREAD_COLOR)
    image = cv2.resize(image, (200, 200))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    kmeans_cluster(image)


def kmeans_cluster(image, K=4):
    Z = np.float32(image.reshape((-1, 3)))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)


    unique, counts = np.unique(label, return_counts=True)
    # print(counts)
    # print(center)

    dst = center[label.flatten()]
    dst = dst.reshape(image.shape)
    dst = np.array(dst, dtype='int16')
    show(dst)
    dic = {}
    for i in unique:
        dic[i] = {'color': center[i - 1], "count": counts[i - 1]}

    pprint.pprint(dic)


def show(image):
    plt.imshow(image)
    plt.show()


preprocess()
