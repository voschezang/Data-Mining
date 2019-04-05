import pandas as pd
import re
from dateutil.parser import parse


def clean_studies(dataF):
    studies = (dataF.ix[:, 'What programme are you in?']).str.upper()
    # various spellings of same study taken together
    studies[studies.str.contains("COMPUTATIONAL SCIENCE")] = "CS"
    studies[studies.str.contains("COMPUTER SCIENCE")] = "CS"
    studies[studies.str.contains("BIOINF")] = "BIOINFORMATICS"
    studies[studies.str.contains("ECONOMETRICS")] = "ECONOMETRICS"
    studies[studies.str.contains("ARTIFICIAL INTELLIGENCE")] = "AI"
    studies[studies.str.contains("BUSINESS ADMINISTRATION")] = "BA"
    studies[studies.str.contains("BUSINESS ANALYTICS")] = "BA"
    return studies


def clean_birthdates(dataF):
    birthDates = dataF.loc[:, "When is your birthday (date)?"]
    birthFrame = pd.DataFrame(index=dataF.index, columns=[
                              'day', 'month', 'year'])

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
                pass  # do nothing

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
    return bfn


def clean_nneigh(dataF):
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
    return nneigh


def clean_money(dataF):
    # get money, replace textual and 100/471 and replace missing by median
    moneyQ = dataF.iloc[:, [11]].squeeze()
    moneyQa = moneyQ.str.replace(",", ".")
    moneyQa = moneyQa.str.replace("€", "")
    moneyQa = moneyQa.str.replace("Barkie.*", "100")
    moneyQa = moneyQa.str.replace(".*[/]{1}.*", "0.21")
    nummoneyQa = moneyQa.apply(pd.to_numeric, errors='coerce')
    nummoneyQa[nummoneyQa.isna()] = nummoneyQa.median()
    return moneyQa, nummoneyQa

def clean_stress_levels(studentinfo):
    # make list containing all numbers of string
    stress_levels = (studentinfo['What is your stress level (0-100)?'])
    for value in stress_levels:
        numbers = []
        for c in value:
            if c.isdigit():
                numbers.append(c)
            if c == ',':
                break
            if c == '-':
                numbers = [0]
                break
        
        # when there are no numbers inside the string assuma value = 50
        if numbers == []:
            numbers = [5, 0]
            value = int(''.join(map(str, numbers)))
        else:
            value = int(''.join(map(str, numbers)))

        # if value is over 100, put value to 100
        if value > 100:
            value = 100
        return value

    # put each value in stress level data to 0-100
    for value in stress_levels:
        
        # check if number is 'Nan'
        if type(value) is float:
            value = 100
        # if number is in string extract stress level
        elif type(value) is str:
            value = clean_stress_levels(value)

def clean_chocolate(dataF):
    field = 'Chocolate makes you.....'
    keys = ['Slim', 'Fat', 'Neither', 'Unknown']
    keys_old = ['SLIM', 'FAT', 'NEITHER',
                'I HAVE NO IDEA WHAT YOU ARE TALKING ABOUT']
    dataF[field] = clean_experience_field(
        dataF[field], keys_old, keys)


def clean_experience(studentinfo):
    field = 'Have you taken a course on statistics?'
    keys = ['Yes', 'No', 'Unknown']
    keys_old = ['MU', 'SIGMA', 'UNKNOWN']
    studentinfo[field] = clean_experience_field(
        studentinfo[field], keys_old, keys)

    field = 'Have you taken a course on machine learning?'
    keys = ['Yes', 'No', 'Unknown']
    keys_old = [x.upper() for x in keys]
    studentinfo[field] = clean_experience_field(
        studentinfo[field], keys_old, keys)

    field = 'Have you taken a course on databases?'
    keys = ['Yes', 'No', 'Unknown']
    keys_old = ["JA", "NEE", "UNKNOWN"]
    studentinfo[field] = clean_experience_field(
        studentinfo[field], keys_old, keys)

    field = 'Have you taken a course on information retrieval?'
    keys = ['Yes', 'No', 'Unknown']
    keys_old = ['1', '0', 'UNKNOWN']
    studentinfo[field] = clean_experience_field(
        studentinfo[field], keys_old, keys)


def clean_experience_field(data, keys_old, keys):
    data = data.str.upper()
    data[~data.isin(keys_old)] = keys[-1]
    for k1, k2 in zip(keys_old, keys):
        data[data.str.contains(k1)] = k2
    return data


