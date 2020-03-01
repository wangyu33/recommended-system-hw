#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : titanic_tpot.py
# Author: WangYu
# Date  : 2020-03-01

# 使用TPOT自动机器学习工具对MNIST进行分类
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from action import *
import numpy as np

# 加载数据
train_data, test_data = load_data()
train_data, test_data = data_fillna(train_data, test_data)
train_data, test_data = data_process(train_data, test_data)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'My']
train_data['My'] = train_data['Age'] + train_data['Sex']
test_data['My'] = test_data['Age'] + test_data['Sex']

train_labels = train_data['Survived']
train_features = train_data[features]
train_x, train_y, label_x, label_y = train_test_split(train_features, train_labels, test_size=0.3, random_state=1)
test_features = test_data[features]

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpot.fit(train_x, label_x)
print(tpot.score(train_y, label_y))
tpot.export('tpot_mnist_pipeline.py')

# output
# Best pipeline: RandomForestClassifier(input_matrix, bootstrap=True, criterion=entropy, max_features=0.7000000000000001, min_samples_leaf=6, min_samples_split=9, n_estimators=100)
# 0.7761194029850746