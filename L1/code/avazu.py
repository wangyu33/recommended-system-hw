#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : avazu.py
# Author: WangYu
# Date  : 2020/1/17

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def celoss(target, predict):
    #交叉熵函数
    target = np.array(target)
    predict = np.array(predict)
    return -(target * np.log(predict) + (1 - target) * np.log(1 - predict)).mean()


def main():
    filename = r'D:\Recommended system\L1\data\avazu-ctr-prediction\train.csv'
    train_data = pd.read_csv(filename, nrows = 300000)
    print(train_data.columns)
    '''
    ['id', 'click', 'hour', 'C1', 'banner_pos'广告, 'site_id' 网站id, 'site_domain' 网站域名,
       'site_category' 网站类别, 'app_id' 软件属性, 'app_domain' 软件域名, 'app_category'软件类别, 'device_id' 设备号,
       'device_ip' 设备id, 'device_model' 设备模型, 'device_type'设备类型, 'device_conn_type', 'C14',
       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'],
    '''
    train_data = train_data.drop(['id'], axis = 1)
    #print(train_data['click'].value_counts())
    for item in train_data.drop(['click'], axis = 1):
        train_data[item] = pd.factorize(train_data[item])[0]
    train, test = train_test_split(train_data, test_size = 0.2, random_state = 0)
    train_x = train.drop(['click'], axis = 1)
    train_y = train['click']
    test_x = test.drop(['click'], axis = 1)
    test_y = test['click']

    # 逻辑回归
    # LR = LogisticRegression(C = 1.0, penalty = 'l1', tol = 1e-6)
    # LR.fit(train_x, train_y)
    # print(LR)
    # predicted = LR.predict_proba(test_x)[:,1]
    # # print(metrics.classification_report(test_y, predicted))
    # # print(metrics.confusion_matrix(test_y, predicted))
    # print("LR 交叉熵：%0.6lf" % celoss(test_y, predicted))

    # LGBM
    lgbm = lgb.LGBMClassifier(silent = False)
    lgbm.fit(train_x, train_y, verbose = 5)
    predicted = lgbm.predict_proba(test_x)[:,1]
    print("lgbm 交叉熵：%0.6lf" % celoss(test_y, predicted))


if __name__ == '__main__':
    main()