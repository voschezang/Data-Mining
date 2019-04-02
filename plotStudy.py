# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:58:33 2019

@author: Gillis
"""
import csv
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import collections as ct

dataF = pd.read_csv('C:\\Users\\Gillis\\Documents\\Uni\\Master2\\DMT\\Data-Mining\\ODI-2019-clean.csv', sep=';')
studs = dataF["study"]
studyCounts = ct.Counter(studs)
studyCounts['other']= 42
concStudyCounts = studyCounts.most_common(11)

df = pd.DataFrame.from_dict(dict(concStudyCounts), orient='index')
ts = df.iloc[[3],:]
df = df.drop("other")
df = df.append(ts)
df = df.rename(index=str, columns={0:"count"})
df = df.rename({"MASTER DIGITAL BUSINESS AND INNOVATION":"DBI"},axis='index')
df.plot(kind='bar')

nneigh = dataF["neighbours"]
#neighc = ct.Counter(nneigh)
#nneigh.plot(kind='bar')
#df = pd.DataFrame.from_dict(neighc, orient = 'index')
#df.plot(kind='bar')
nneigh[nneigh>470] = 470
plt.hist(nneigh, bins =100,rwidth=0.85)
plt.hist(nneigh[nneigh<11],rwidth=0.85)

sett = dataF["moneyV"]
sett[sett>100] = 100
plt.hist(sett, bins =100)
plt.hist(sett[sett<10],rwidth=0.85)