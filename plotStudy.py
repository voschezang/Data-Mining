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
