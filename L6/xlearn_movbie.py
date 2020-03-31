#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : xlearn_movbie.py
# Author: WangYu
# Date  : 2020-03-31

import xlearn as xl
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class FFMFormatPandas:
    def __init__(self):
        self.field_index_ = None
        self.feature_index_ = None
        self.y = None

    def fit(self, df, y=None):
        self.y = y
        df_ffm = df[df.columns.difference([self.y])]
        if self.field_index_ is None:
            self.field_index_ = {col: i for i, col in enumerate(df_ffm)}

        if self.feature_index_ is not None:
            last_idx = max(list(self.feature_index_.values()))

        if self.feature_index_ is None:
            self.feature_index_ = dict()
            last_idx = 0

        for col in df.columns:
            vals = df[col].unique()
            for val in vals:
                if pd.isnull(val):
                    continue
                name = '{}_{}'.format(col, val)
                if name not in self.feature_index_:
                    self.feature_index_[name] = last_idx
                    last_idx += 1
            self.feature_index_[col] = last_idx
            last_idx += 1
        return self

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        return self.transform(df)

    def transform_row_(self, row, t):
        ffm = []
        if self.y != None:
            ffm.append(str(row.loc[row.index == self.y][0]))
        if self.y is None:
            ffm.append(str(0))

        for col, val in row.loc[row.index != self.y].to_dict().items():
            col_type = t[col]
            name = '{}_{}'.format(col, val)
            if col_type.kind ==  'O':
                ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
            elif col_type.kind == 'i':
                ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], val))
        return ' '.join(ffm)

    def transform(self, df):
        t = df.dtypes.to_dict()
        return pd.Series({idx: self.transform_row_(row, t) for idx, row in df.iterrows()})

def to_txt(m, filename):
    with open(filename, 'w') as f:
        for temp in m:
            temp = temp + '\n'
            f.write(temp)

# 整合数据
# # ratings
# rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
# rdtype = {'user_id': np.int64, 'movieId': np.int64, 'rating': np.int64, 'timestamp':np.int64}
# ratings = pd.read_csv('ratings.dat', sep = '::', names = rnames, engine = 'python', dtype = rdtype)
# ratings.info()
#
# # movie
# mnames = ['movie_id', 'title', 'genres']
# movies = pd.read_csv('movies.dat',sep='::', names=mnames, engine='python')
# movies.info()
#
# # user
# unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
# users = pd.read_csv('users.dat',sep='::', names=unames, engine='python')
# users.info()
#
# temp = pd.merge(ratings, movies, on = 'movie_id')
# data = pd.merge(temp, users, on = 'user_id')
# data.to_csv('movieslen.csv', index = None)

data = pd.read_csv('movielens_sample.txt')
features = ["movie_id", "user_id", "age", "gender","genres","occupation", "zip"] #, "rating"
data.info()
for index in features:
    enc = LabelEncoder()
    data[index] = enc.fit_transform(data[index])

rating = data['rating']
data = data.drop(['title'], axis = 1)
train_x, train_y, label_x, label_y = train_test_split(data, rating, test_size = 0.5)

csv2lib = FFMFormatPandas()
train_x = csv2lib.fit_transform(train_x, y = 'rating')
train_y = csv2lib.fit_transform(train_y, y = 'rating')
to_txt(train_x, "small_train.txt")
to_txt(train_y, "small_test.txt")

ffm_model = xl.create_fm()
# 设置训练集和测试集
ffm_model.setTrain("./small_train.txt")
ffm_model.setValidate("./small_test.txt")

# 设置参数，任务为二分类，学习率0.2，正则项lambda: 0.002，评估指标 accuracy
param = {'task':'reg', 'lr':0.2, 'lambda':0.002, 'metric':'acc', 'opt' : 'sgd'}

# FFM训练，并输出模型
ffm_model.fit(param, './model.out')

# 设置测试集，将输出结果转换为0-5
ffm_model.setTest("./small_test.txt")
# 使用训练好的FFM模型进行预测，输出到output.txt
ffm_model.predict("./model.out", "./output.txt")


