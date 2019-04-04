# -*- coding: utf-8 -*-
"""
Cleans part of ODI file on various questions
Cleans study, birthdates, number of neighbours and money expected to return
Nans in neighbours and money expected are replace by median, others left empty
"""

import csv
import pandas as pd
import re
from dateutil.parser import parse

dataF = pd.read_csv('ODI-2019-csv.csv', sep=';')
# dataF = pd.read_csv('C:\\Users\\Gillis\\Documents\\Uni\\Master2\\DMT\\Data-Mining\\ODI-2019-csv.csv', sep=';')
studies = (dataF.ix[:, 'What programme are you in?']).str.upper()
# various spellings of same study taken together
studies[studies.str.contains("COMPUTATIONAL SCIENCE")] = "CS"
studies[studies.str.contains("COMPUTER SCIENCE")] = "CS"
studies[studies.str.contains("BIOINF")] = "BIOINFORMATICS"
studies[studies.str.contains("ECONOMETRICS")] = "ECONOMETRICS"
studies[studies.str.contains("ARTIFICIAL INTELLIGENCE")] = "AI"
studies[studies.str.contains("BUSINESS ADMINISTRATION")] = "BA"
studies[studies.str.contains("BUSINESS ANALYTICS")] = "BA"

birthDates = dataF.loc[:, "When is your birthday (date)?"]
birthFrame = pd.DataFrame(index=dataF.index, columns=['day', 'month', 'year'])


# for various regex patterns, split dates into day/month/year
dateP = re.compile("^[0-9]{2}[./-]{1}|[0-9]{2}[./-]{1}[0-9]{4}")
check = birthDates.str.contains(dateP)
for i in range(birthDates.size):
    if check[i]:
        birthFrame.iloc[[i], [0]] = birthDates[i][0:2]
        birthFrame.iloc[[i], [1]] = birthDates[i][3:5]
        birthFrame.iloc[[i], [2]] = birthDates[i][6:10]

dateP = re.compile("^[0-9]{1}[./-]{1}[0-9]{2}[./-]{1}[0-9]{4}$")
check = birthDates.str.contains(dateP)
for i in range(birthDates.size):
    if check[i]:
        birthFrame.iloc[[i], [0]] = birthDates[i][0:1]
        birthFrame.iloc[[i], [1]] = birthDates[i][2:4]
        birthFrame.iloc[[i], [2]] = birthDates[i][5:9]

dateP = re.compile("^[0-9]{1}[./-]{1}[0-9]{1}[./-]{1}[0-9]{4}$")
check = birthDates.str.contains(dateP)
for i in range(birthDates.size):
    if check[i]:
        birthFrame.iloc[[i], [0]] = birthDates[i][0:1]
        birthFrame.iloc[[i], [1]] = birthDates[i][2:3]
        birthFrame.iloc[[i], [2]] = birthDates[i][4:8]

dateP = re.compile("^[0-9]{2}[./-]{1}[0-9]{1}[./-]{1}[0-9]{4}$")
check = birthDates.str.contains(dateP)
for i in range(birthDates.size):
    if check[i]:
        birthFrame.iloc[[i], [0]] = birthDates[i][0:2]
        birthFrame.iloc[[i], [1]] = birthDates[i][3:4]
        birthFrame.iloc[[i], [2]] = birthDates[i][5:9]

dateP = re.compile("^[0-9]{2}[./-]{1}[0-9]{2}[./-]{1}[0-9]{2}$")
check = birthDates.str.contains(dateP)

for i in range(birthDates.size):
    if check[i]:
        birthFrame.iloc[[i], [0]] = birthDates[i][0:2]
        birthFrame.iloc[[i], [1]] = birthDates[i][3:5]
        birthFrame.iloc[[i], [2]] = birthDates[i][6:8]

dateP = re.compile("^[0-9]{1}[./-]{1}[0-9]{2}[./-]{1}[0-9]{2}$")
check = birthDates.str.contains(dateP)
for i in range(birthDates.size):
    if check[i]:
        birthFrame.iloc[[i], [0]] = birthDates[i][0:1]
        birthFrame.iloc[[i], [1]] = birthDates[i][2:4]
        birthFrame.iloc[[i], [2]] = birthDates[i][5:7]

dateP = re.compile("^[0-9]{1}[./-]{1}[0-9]{1}[./-]{1}[0-9]{2}$")
check = birthDates.str.contains(dateP)
for i in range(birthDates.size):
    if check[i]:
        birthFrame.iloc[[i], [0]] = birthDates[i][0:1]
        birthFrame.iloc[[i], [1]] = birthDates[i][2:3]
        birthFrame.iloc[[i], [2]] = birthDates[i][4:6]

dateP = re.compile("^[0-9]{2}[./-]{1}[0-9]{1}[./-]{1}[0-9]{2}$")
check = birthDates.str.contains(dateP)
for i in range(birthDates.size):
    if check[i]:
        birthFrame.iloc[[i], [0]] = birthDates[i][0:2]
        birthFrame.iloc[[i], [1]] = birthDates[i][3:4]
        birthFrame.iloc[[i], [2]] = birthDates[i][5:7]

dateP = re.compile("^[0-9]{8}$")
check = birthDates.str.contains(dateP)
for i in range(birthDates.size):
    if check[i]:
        birthFrame.iloc[[i], [0]] = birthDates[i][0:2]
        birthFrame.iloc[[i], [1]] = birthDates[i][2:4]
        birthFrame.iloc[[i], [2]] = birthDates[i][4:8]

for i in range(birthDates.size):
    if birthFrame.iloc[[i], [0]].isna().squeeze():
        try:
            ymd = parse(birthDates[i], fuzzy=True)
            birthFrame.iloc[[i], [0]] = ymd.day
            birthFrame.iloc[[i], [1]] = ymd.month
            birthFrame.iloc[[i], [2]] = ymd.year
            if ymd.year == 2019:
                birthFrame.iloc[[i], [2]] = ""
        except:
            print("io")
            # do nothing


# make into numerics, change '95 into 1995, swap MM-DD-YYYY to DD-MM-YYYY if obvious
bfn = pd.DataFrame(index=dataF.index, columns=['day', 'month', 'year'])
bfn = birthFrame.apply(pd.to_numeric, errors='coerce')
for i in range(int(bfn.size / 3)):
    if (bfn.iloc[[i], [2]] < 100).bool():
        bfn.iloc[[i], [2]] = bfn.iloc[[i], [2]] + 1900
    if (bfn.iloc[[i], [1]] > 12).bool():
        day = bfn.iloc[[i], [1]]
        bfn.iloc[[i], [1]] = bfn.iloc[[i], [0]]
        bfn.iloc[[i], [0]] = day

# rename columns
bfn.columns = ['Day', 'Month', 'Year']

# get number of neighbours, replace missing by median
nneigh = dataF.iloc[:, [9]]
nneighNum = nneigh.apply(pd.to_numeric, errors='coerce')
checkNull = nneighNum.isna().T.squeeze()
for i in range(nneighNum.size):
    if (checkNull[i]):
        if (re.search(re.compile("[0-9]+"), nneigh.iloc[[i], [0]].squeeze()) != None):
            nneighNum.iloc[[i], [0]] = re.search(re.compile(
                "[0-9]+"), nneigh.iloc[[i], [0]].squeeze())[0]
        else:
            nneighNum.iloc[[i], [0]] = nneighNum.median().squeeze()
            print("sdf")

# get money, replace textual and 100/471 and replace missing by median
moneyQ = dataF.iloc[:, [11]].squeeze()
moneyQa = moneyQ.str.replace(",", ".")
moneyQa = moneyQa.str.replace("€", "")
moneyQa = moneyQa.str.replace("Barkie.*", "100")
moneyQa = moneyQa.str.replace(".*[/]{1}.*", "0.21")
nummoneyQa = moneyQa.apply(pd.to_numeric, errors='coerce')
nummoneyQa[nummoneyQa.isna()] = nummoneyQa.median()

# create df with new data
studyV = studies
birthMat = bfn
neighbourV = nneighNum
moneyV = nummoneyQa

dataNew = dataF
dataNew = dataNew.drop(dataNew.columns[11], axis=1)
dataNew = dataNew.drop(["What programme are you in?", "When is your birthday (date)?",
                        "Number of neighbors sitting around you?"], axis=1)
dataNew["Program"] = studyV
dataNew = dataNew.join(birthMat)
dataNew["Neighbours"] = neighbourV
dataNew["Money"] = nummoneyQa

# rename remaining columns
keys = {'What programme are you in?': 'Program',
        'Have you taken a course on machine learning?': 'ML',
        'Have you taken a course on information retrieval?': 'IR',
        'Have you taken a course on statistics?': 'Stat',
        'Have you taken a course on databases?': 'DB',
        'What is your gender?': 'Gender',
        'Chocolate makes you.....': 'Chocolate',
        'Did you stand up?': 'Stand up',
        'Give a random number': 'Rand',
        'Time you went to be Yesterday': 'Bedtime',
        'What makes a good day for you (1)?': 'Good day (1)',
        'What makes a good day for you (2)?': 'Good day (2)',
        'What is your stress level (0-100)?': 'Stress level'}
dataNew.rename(index=str, columns=keys, inplace=True)

print('done')
print(dataNew.keys())
if __name__ == "__main__":
    dataNew.to_csv('ODI-2019-clean.csv', sep=';')
