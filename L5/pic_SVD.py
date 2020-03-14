#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : pic_SVD.py
# Author: WangYu
# Date  : 2020-03-14

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

pic = cv2.imread('3.jpg', cv2.IMREAD_COLOR)
gray_img = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
print(gray_img.shape)

#原图
plt.subplot(2,2,1)
plt.imshow(gray_img, cmap = 'gray')
plt.title('original pic')

# 对图像矩阵A进行奇异值分解，得到p,s,q
p,s,q = svd(gray_img, full_matrices=False)

#压缩比
pic_size = gray_img.shape[1]
s_1 = np.zeros([pic_size, pic_size])
s_10 = np.zeros([pic_size, pic_size])
s_50 = np.zeros([pic_size,pic_size])
for i in range(int(0.01 * pic_size)):
    s_1[i][i] = s[i]
for i in range(int(0.1 * pic_size)):
    s_10[i][i] = s[i]
for i in range(int(0.5 * pic_size)):
    s_50[i][i] = s[i]

#绘图
plt.subplot(2,2,2)
plt.imshow(p.dot(s_1).dot(q), cmap = 'gray')
plt.title('compression ratio = 1%')

plt.subplot(2,2,3)
plt.imshow(p.dot(s_10).dot(q), cmap = 'gray')
plt.title('compression ratio = 10%')

plt.subplot(2,2,4)
plt.imshow(p.dot(s_50).dot(q), cmap = 'gray')
plt.title('compression ratio = 50%')
