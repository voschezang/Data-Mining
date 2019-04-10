# -*- coding: utf-8 -*-
"""
Cleans ODI
Nans in neighbours and money expected are replace by median, others left empty
"""
import pandas as pd
from odi import functions
from util.bedtime import parse_bedtimes

dataF = pd.read_csv('ODI-2019-csv.csv', sep=';')
# dataF = pd.read_csv('C:\\Users\\Gillis\\Documents\\Uni\\Master2\\DMT\\Data-Mining\\ODI-2019-csv.csv', sep=';')
# dataF.replace([np.inf, -np.inf], np.nan)
moneyQa, nummoneyQa = functions.clean_money(dataF)


dataF['What is your stress level (0-100)?'] = functions.clean_stress_level(
    dataF['What is your stress level (0-100)?'])


# clean rest
functions.clean_experience(dataF)
functions.clean_chocolate(dataF)

# create df with new data
studyV = functions.clean_studies(dataF)
birthMat = functions.clean_birthdates(dataF)
neighbourV = functions.clean_nneigh(dataF)
moneyV = moneyQa  # TODO unused?

dataNew = dataF
dataNew = dataNew.drop(dataNew.columns[11], axis=1)
dataNew = dataNew.drop(["What programme are you in?",
                        "When is your birthday (date)?",
                        "Number of neighbors sitting around you?",
                        "Time you went to be Yesterday"], axis=1)
dataNew["Programme"] = studyV
dataNew = dataNew.join(birthMat)
dataNew["Neighbours"] = neighbourV
dataNew["Money"] = nummoneyQa
dataNew["Bedtime"] = parse_bedtimes(dataF['Time you went to be Yesterday'])


# rename remaining columns
keys = {'What programme are you in?': 'Program',
        'Have you taken a course on machine learning?': 'ML',
        'Have you taken a course on information retrieval?': 'IR',
        'Have you taken a course on statistics?': 'Stat',
        'Have you taken a course on databases?': 'DB',
        'What is your gender?': 'Gender',
        'Chocolate makes you.....': 'Chocolate',
        'Did you stand up?': 'Stand Up',
        'Give a random number': 'Rand',
        # 'Time you went to be Yesterday': 'Bedtime',
        'What makes a good day for you (1)?': 'Good day (1)',
        'What makes a good day for you (2)?': 'Good day (2)',
        'What is your stress level (0-100)?': 'Stress level'}
dataNew.rename(index=str, columns=keys, inplace=True)

print('done')
print(dataNew.keys())
if __name__ == "__main__":
    dataNew.to_csv('ODI-2019-clean.csv', sep=';')
