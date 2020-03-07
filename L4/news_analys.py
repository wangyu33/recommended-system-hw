#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : news_analys.py
# Author: WangYu
# Date  : 2020-03-07

from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import jieba

# 去掉停用词
def remove_stop_words(f):
	stop_words = ['今天','各','要']
	for stop_word in stop_words:
		f = f.replace(stop_word, '')
	return f

# 生成词云
def create_word_cloud(f):
	print('根据词频，开始生成词云!')
	f = remove_stop_words(f)
	cut_text = jieba.cut(f)
	#print(cut_text)
	cut_text = " ".join(cut_text)
	wc = WordCloud(
		font_path="simhei.ttf",
		max_words=100,
		width=2000,
		height=1200,
    )
	wordcloud = wc.generate(cut_text)
	# 写词云图片
	wordcloud.to_file("wordcloud.jpg")
	# 显示词云文件
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.show()

# 数据加载
f = open('news.txt', 'r', encoding = 'gbk', errors = 'ignore')
text = f.read()
# 生成词云
create_word_cloud(text)