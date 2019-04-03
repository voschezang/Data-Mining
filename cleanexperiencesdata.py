import csv
import pandas as pd
import numpy as np
from dateutil.parser import parser

studentinfo = pd.read_csv('ODI-2019-csv.csv', sep=';')
# studentinfo.describe()

stat_exp = (studentinfo['Have you taken a course on statistics?']).str.upper()
stat_exp[stat_exp.str.contains("MU")] = 'YES'
stat_exp[stat_exp.str.contains("SIGMA")] = 'NO'
studentinfo['Have you taken a course on statistics?'] = stat_exp

# check if machine learning experience values are all valid
ml_exp = (studentinfo['Have you taken a course on machine learning?']).str.upper()
for row in ml_exp:
	if row != 'YES' and row != 'NO' and row != 'UNKNOWN':
		row = 'UNKNOWN'
studentinfo['Have you taken a course on machine learning?'] = ml_exp

# check if information retrieval experience values are all valid
ir_exp = (studentinfo['Have you taken a course on information retrieval?']).str.upper()
for row in ir_exp:
	if row != 1 and row != 0 and row != 'UNKNOWN':
		row = 'UNKNOWN'
studentinfo['Have you taken a course on information retrieval?'] = ir_exp

# change databases experience values to YES/NO/UNKOWN
db_exp = (studentinfo['Have you taken a course on databases?']).str.upper()
db_exp[db_exp.str.contains("JA")] = 'YES'
db_exp[db_exp.str.contains("NEE")] = 'NO'
for row in db_exp:
	if row != "YES" and row != "NO" and row != 'UNKNOWN':
		row = 'UNKNOWN'
studentinfo['Have you taken a course on databases?'] = db_exp





