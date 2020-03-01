#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : action.py
# Author: WangYu
# Date  : 2020-03-01

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model.logistic import LogisticRegression

def load_data():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    print(train_data.columns)
    # Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
    #        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
    #       dtype='object')
    print('查看数据信息：列名、非空个数、类型等')
    print(train_data.info())
    print('-' * 30)
    print('查看数据摘要')
    print(train_data.describe())
    print('-' * 30)
    print('查看离散数据分布')
    print(train_data.describe(include=['O']))
    return train_data, test_data

def data_fillna(train_data, test_data):
    # 使用平均年龄来填充年龄中的nan值
    train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
    test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
    # 使用票价的均值填充票价中的nan值
    train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
    test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

    print(train_data['Embarked'].value_counts())
    train_data['Embarked'].fillna('S', inplace=True)
    test_data['Embarked'].fillna('S', inplace=True)
    return train_data, test_data

def data_process(train_data, test_data):
    #性别
    train_data['Sex'] = train_data['Sex'].map(lambda x: 1 if x == 'female' else 0)
    test_data['Sex'] = test_data['Sex'].map(lambda x: 1 if x == 'female' else 0)

    #年龄
    train_data.loc[train_data['Age']  <= 18, 'Age'] = 0
    train_data.loc[(train_data['Age'] > 18) & (train_data['Age'] <= 32), 'Age'] = 1
    train_data.loc[(train_data['Age'] > 32) & (train_data['Age'] <= 46), 'Age'] = 2
    train_data.loc[(train_data['Age'] > 46) & (train_data['Age'] <= 60), 'Age'] = 3
    train_data.loc[train_data['Age'] > 60, 'Age'] = 4

    test_data.loc[test_data['Age'] <= 18, 'Age'] = 0
    test_data.loc[(test_data['Age'] > 18) & (test_data['Age'] <= 32), 'Age'] = 1
    test_data.loc[(test_data['Age'] > 32) & (test_data['Age'] <= 46), 'Age'] = 2
    test_data.loc[(test_data['Age'] > 46) & (test_data['Age'] <= 60), 'Age'] = 3
    test_data.loc[test_data['Age'] > 60, 'Age'] = 4

    #费用
    train_data.loc[train_data['Fare'] <= 7.91, 'Fare'] = 0
    train_data.loc[(train_data['Fare'] > 7.91) & (train_data['Fare'] <= 14.454), 'Fare'] = 1
    train_data.loc[(train_data['Fare'] > 14.454) & (train_data['Fare'] <= 31), 'Fare'] = 2
    train_data.loc[train_data['Fare'] > 31, 'Fare'] = 3

    test_data.loc[test_data['Fare'] <= 7.91, 'Fare'] = 0
    test_data.loc[(test_data['Fare'] > 7.91) & (test_data['Fare'] <= 14.454), 'Fare'] = 1
    test_data.loc[(test_data['Fare'] > 14.454) & (test_data['Fare'] <= 31), 'Fare'] = 2
    test_data.loc[test_data['Fare'] > 31, 'Fare'] = 3

    #港口
    dict = {'S': 0, 'C':1, 'Q': 2}
    train_data['Embarked'] = train_data['Embarked'].map(lambda x: dict[x])
    test_data['Embarked'] = test_data['Embarked'].map(lambda x: dict[x])
    return train_data, test_data

def main():
    train_data, test_data = load_data()
    train_data, test_data = data_fillna(train_data, test_data)
    train_data, test_data = data_process(train_data, test_data)
    #特征选择
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'My']
    train_data['My'] = train_data['Age'] + train_data['Sex']
    test_data['My'] = test_data['Age'] + test_data['Sex']

    train_labels = train_data['Survived']
    train_features = train_data[features]
    train_x, train_y, label_x, label_y = train_test_split(train_features, train_labels, test_size=0.3, random_state=1)
    test_features = test_data[features]

    LR = LogisticRegression(max_iter=100, verbose=True, random_state=33, tol=1e-4)
    LR.fit(train_x, label_x)
    predict = LR.predict_proba(train_y)[:, 1]
    feature_importance = LR.coef_[0]
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    print('feature importance is:')
    print(feature_importance)
    print("LR auc：%0.6lf" % metrics.roc_auc_score(label_y, predict))

    SVM = SVC(kernel='rbf', probability = True, C=0.2)
    SVM.fit(train_x, label_x)
    predict_svm = SVM.predict_proba(train_y)[:, 1]
    print("SVM auc：%0.6lf" % metrics.roc_auc_score(label_y, predict_svm))

    LGB = lgb.LGBMClassifier(boosting_type='gbdt',
                             objective='binary',
                             metric='auc',
                             verbose=0,

                             learning_rate = 0.01,
                             num_leaves = 31,
                             feature_fraction = 0.8,
                             bagging_fraction = 0.8,
                             bagging_freq = 2,
                             lambda_l1 = 0.8,
                             lambda_l2 = 0,
                             max_depth = 5,
                             # silent = False
                             cat_smooth = 1
                             )
    LGB.fit(train_x, label_x)
    predict_LGB = LGB.predict_proba(train_y)[:, 1]
    lgb.plot_importance(LGB, max_num_features = 30)
    print("LGB auc：%0.6lf" % metrics.roc_auc_score(label_y, predict))

if __name__ == '__main__':
    main()

