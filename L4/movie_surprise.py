#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : movie_surprise.py
# Author: WangYu
# Date  : 2020-03-07

from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise import BaselineOnly
from surprise.model_selection import KFold
from surprise.model_selection import GridSearchCV
import pandas as pd
# 数据读取
# sep分隔符，line_format行的形式 skip_lines 跳过的行数

def data_process():
    # 建立名字序号的dict,以及查看总共电影类别
    data = pd.read_csv('movies.csv', encoding = 'unicode_escape')
    rid_to_name = {}
    name_to_rid = {}
    category = set()
    for i in range(len(data['movieId'])):
        rid_to_name[data['movieId'][i]] = data['title'][i]
        name_to_rid[data['title'][i]] = data['movieId'][i]
        cat = data['genres'][i].split('|')
        for temp in cat:
            category.add(temp)
    return rid_to_name, name_to_rid,category

def main():
    rid_to_name, name_to_rid, category = data_process()
    size = len(category)
    print(size)     #20类
    # 数据读取
    reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    data = Dataset.load_from_file('./ratings.csv', reader=reader)
    train_set = data.build_full_trainset()

    #训练
    # # 使用ALS 相互迭代，求user矩阵和item矩阵
    # bsl_options = {'method': 'als', 'n_epochs': [5], 'reg_u': [10], 'reg_i': [10]}
    # # 使用sgd 沿着最大梯度下降
    # #bsl_options = {'method': 'sgd', 'n_epochs': [5, 10], 'reg':[0.01,0.02]}
    # gs = GridSearchCV(BaselineOnly(bsl_options = bsl_options), bsl_options, measures = ['rmse'], cv = 3)
    # print('baseline begin')
    # gs.fit(data)
    # print(gs.best_score['rmse'])
    # print(gs.best_params['rmse'])

    #SVD SVD矩阵分解
    para_grid = {'n_epochs': [5],
                 'lr_all': [0.002],
                 'reg_all': [0.4]}
    gs = GridSearchCV(SVD, para_grid, measures=['rmse'], cv=3)
    print('SVD begin')
    gs.fit(data)
    print(gs.best_score['rmse'])
    print(gs.best_params['rmse'])

    model = gs.best_estimator['rmse']
    model.fit(train_set)
    uid = str(196)
    iid = str(302)
    # 输出uid对iid的预测结果
    pred = model.predict(uid, iid, r_ui=4, verbose=True)
    print(pred)

if __name__ == '__main__':
    main()


