#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : dimission_predict.py
# Author: WangYu
# Date  : 2020/1/24

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import catboost as cat
import ngboost as ng
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

def main():
    train_0 = pd.read_csv('train.csv')
    test_0 = pd.read_csv('test.csv')
    #print(train_0.head(10))

    header = ['BusinessTravel', 'Department','EducationField',
       'Gender','JobRole','MaritalStatus','OverTime']

    # 删除无用特征
    user_id = test_0['user_id']
    train_0 = train_0.drop(['user_id','EmployeeCount','Over18'], axis = 1)
    test_0 = test_0.drop(['user_id','EmployeeCount','Over18'], axis = 1)

    #特征编码
    for index in header:
        LE = LabelEncoder()
        train_0[index] = LE.fit_transform(train_0[index])
        test_0[index] = LE.transform(test_0[index])
    LE = LabelEncoder()
    label_0 =LE.fit_transform(train_0['Attrition'])
    train_0 = train_0.drop(['Attrition'], axis = 1)
    train_x, train_y, label_x, label_y = train_test_split(train_0,label_0, test_size = 0.3, random_state = 1)
    # 标准化

    # LGBM 调参

    parameters = {
        'max_depth': [15, 20, 25],
        'learning_rate': [0.01, 0.05],
        'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
        'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
        'bagging_freq': [2, 4, 5, 6, 8],
        'lambda_l1': [0.6, 0.7, 0.8],
        'lambda_l2': [0, 15, 35],
        'cat_smooth': [1, 10, 15]
    }

    LGB = lgb.LGBMClassifier(boosting_type='gbdt',
                            objective = 'binary',
                            metric = 'auc',
                            verbose = 0,


                            learning_rate = 0.01,
                            num_leaves = 35,
                            feature_fraction=0.8,
                            bagging_fraction= 0.7,
                            bagging_freq= 2,
                            lambda_l1= 0.8,
                            lambda_l2= 0,
                            max_depth= 15,
                            #silent = False
                            cat_smooth = 1
                             )
    # gsearch = GridSearchCV(LGB, param_grid=parameters, scoring='roc_auc', cv = 3)
    # gsearch.fit(train_0, label_0)
    #
    # print("Best score: %0.3f" % gsearch.best_score_)
    # print("Best parameters set:")
    # best_parameters = gsearch.best_estimator_.get_params()
    # for param_name in sorted(parameters.keys()):
    #     print("\t%s: %r" % (param_name, best_parameters[param_name]))



    # LGB.fit(train_0, label_0)
    # predict = LGB.predict_proba(test_0)[:,1]
    #
    # test_0['Attrition'] = predict
    # test_0['user_id'] = user_id
    # test_0[['user_id','Attrition']].to_csv('submit_lgb.csv', index = False)

    LGB.fit(train_x, label_x)
    predict = LGB.predict_proba(train_y)[:, 1]
    print("LGB auc：%0.6lf" % metrics.roc_auc_score(label_y, predict))

    SVM = SVC(kernel = 'rbf',probability = True, C = 0.2)
    SVM.fit(train_x, label_x)
    predict_svm = SVM.predict_proba(train_y)[:, 1]
    print("SVM auc：%0.6lf" % metrics.roc_auc_score(label_y, predict_svm))

    CAT = cat.CatBoostClassifier()
    CAT.fit(train_x, label_x)
    predict_svm = CAT.predict_proba(train_y)[:, 1]
    print("cat auc：%0.6lf" % metrics.roc_auc_score(label_y, predict_svm))

    NG= ng.NGBClassifier()
    NG.fit(train_x, label_x)
    predict_ng = NG.pred_dist(train_y)
    predict_ng = predict_ng.probs[1, :]
    print("NG auc：%0.6lf" % metrics.roc_auc_score(label_y, predict_ng))




if __name__ == "__main__":
    main()