#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : KNN_baseline_movie.py
# Author: WangYu
# Date  : 2020-04-12

from surprise import KNNWithMeans
from surprise import Dataset, Reader
from surprise import KNNBaseline
from surprise.model_selection import KFold
from surprise import accuracy

# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('./ratings.csv', reader=reader)

KF = KFold(n_splits=3)
algo = KNNBaseline(k=50, sim_options={'user_based': False, 'verbose': 'True'})

for train, test in KF.split(data):
    algo.fit(train)
    predictions = algo.test(test)
    accuracy.rmse(predictions, verbose=True)
    accuracy.mae(predictions, verbose=True)




