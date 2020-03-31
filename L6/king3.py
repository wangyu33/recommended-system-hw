#!/usr/bin/env python
# -*- coding:utf-8 -*-
# File  : king3.py
# Author: WangYu
# Date  : 2020-03-29

from gensim.models import word2vec
import multiprocessing
import jieba
import os
from utils import files_processing

# 源文件所在目录
source_folder = './source'
segment_folder = './segment'

# 字词分割，对整个文件内容进行字词分割
def segment_lines(file_list,segment_out_dir,stopwords=[]):
    for i,file in enumerate(file_list):
        segment_out_name=os.path.join(segment_out_dir,'segment_{}.txt'.format(i))
        with open(file, 'rb') as f:
            document = f.read()
            document_cut = jieba.cut(document)
            sentence_segment=[]
            for word in document_cut:
                if word not in stopwords:
                    sentence_segment.append(word)
            result = ' '.join(sentence_segment)
            result = result.encode('utf-8')
            with open(segment_out_name, 'wb') as f2:
                f2.write(result)

# 对source中的txt文件进行分词，输出到segment目录中
file_list=files_processing.get_files_list(source_folder, postfix='*.txt')
segment_lines(file_list, segment_folder)

# 如果目录中有多个文件，可以使用PathLineSentences
segment_folder = './segment'
sentences = word2vec.PathLineSentences(segment_folder)

model = word2vec.Word2Vec(sentences, size=128, window=5, min_count=5, workers=multiprocessing.cpu_count())
# 保存模型
model.save('./models/word2Vec.model')
print(model.wv.most_similar('曹操'))
print(model.wv.most_similar(positive=['曹操', '刘备'], negative=['张飞']))









