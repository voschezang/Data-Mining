# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:30:20 2019

@author: familie
"""

#import
import pandas as pd
import sklearn.ensemble as es
import sklearn.model_selection as ms
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#read sms dataset, with specific separator due to ; occuring in texts
dataF = pd.read_csv('smscollection.csv', sep='am;',engine='python',index_col=False,skiprows=1,header=None,names={'hs','text'})

#calculate word occurences and convert to tdidf
vector = CountVectorizer()
word_counts = vector.fit_transform(dataF['text'])
word_tfidf = TfidfTransformer().fit_transform(word_counts)

#convert to spare data frame to include other variables later
sdf = pd.SparseDataFrame(word_tfidf,
                         columns=vector.get_feature_names(), 
                         default_fill_value=0)

#calculate target binary variable
target = dataF['hs']=='sp'

#add more features to dataset
sdf['textLength'] = dataF['text'].str.len()
sdf['Uppercase'] = dataF['text'].str.count(r'[A-Z]')
sdf['pound'] = dataF['text'].str.count(r'£')
sdf['wordNumber'] = dataF['text'].str.split().str.len()

#train random forest and evaluate
rf = es.RandomForestClassifier(n_estimators=100)
rfScore = ms.cross_val_score(rf, sdf, target, scoring='accuracy', cv=5)
print(np.mean(rfScore))
