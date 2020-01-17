#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : minist_cart.py
# Author: WangYu
# Date  : 2020/1/16

import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

def main():
    digit = load_digits()
    data = digit.data
    label = digit.target
    #划分数据集
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size = 0.3, random_state = 0)
    # print(digit.target[0])
    # plt.imshow(digit.images[0],cmap='gray')
    # plt.show()

    # 数据预处理——规范化 standardScaler() 标准化 -均值/方差
    train_x = preprocessing.StandardScaler().fit_transform(train_x)
    test_x = preprocessing.StandardScaler().fit_transform(test_x)

    #训练 CART
    CART = DecisionTreeClassifier()
    CART.fit(train_x, train_y)
    print(CART)
    predicted = CART.predict(test_x)
    print(metrics.classification_report(test_y, predicted))
    print(metrics.confusion_matrix(test_y, predicted))
    print("CART 准确率：%0.6lf" % metrics.accuracy_score(test_y, predicted))

    #训练 逻辑回归
    LR = LogisticRegression()
    LR.fit(train_x, train_y)
    print(LR)
    predicted = LR.predict(test_x)
    # predicted = np.ceil(predicted)
    # predicted = np.where(predicted < 0, 0, predicted)
    # predicted = np.where(predicted > 9, 9, predicted)
    # predicted = np.array(predicted, dtype = int)
    print(metrics.classification_report(test_y, predicted))
    print(metrics.confusion_matrix(test_y, predicted))
    print("LR 准确率：%0.6lf" % metrics.accuracy_score(test_y, predicted))

    # 训练 SVM
    SVM = SVC()
    SVM.fit(train_x, train_y)
    print(SVM)
    predicted = SVM.predict(test_x)
    print(metrics.classification_report(test_y, predicted))
    print(metrics.confusion_matrix(test_y, predicted))
    print("SVM 准确率：%0.6lf" % metrics.accuracy_score(test_y, predicted))

if __name__ == '__main__':
    main()