# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 13:10:27 2018

@author: Justin
"""
import pandas as pd
from gensim.models.word2vec import Word2Vec

#CREATING AND SAVING MODEL
sentenceList = list()
bt = pd.read_table('delicious-data/bookmark_tags.dat',header=0, sep='\s+', index_col=False)
tags = pd.read_table('delicious-data/tags.dat',header=0, sep='\s+', index_col=0, encoding='cp1252')
bookmarks = bt.groupby('bookmarkID')

for name,group in bookmarks:
    sentence = ""
    tagList = bookmarks.get_group(name)["tagID"]
    for tag in tagList:
        #print(tag)
        value = tags.get_value(tag, "value")
        sentence = sentence + value + " "
    sentence = sentence[:-1]
    sentenceList.append(sentence)

print(sentenceList)
model = Word2Vec(sentenceList, size=100, window=5, workers=4)
model.save('model-save')

#LOADING MODEL
model = Word2Vec.load('model-save')
print(model.wv.vocab.keys())