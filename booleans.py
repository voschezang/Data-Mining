import pandas as pd
from matplotlib import rcParams
import pickle
import numpy as np
import phonenumbers
from phonenumbers.phonenumberutil import region_code_for_country_code
import requests
import pycountry

np.random.seed(123)

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14

data = pd.read_csv('data/training_set_VU_DM.csv', sep=',')

print(data.columns)
srch_saturday_night_bool = data['srch_saturday_night_bool']
random_bool = data['random_bool']
click_bool = data['click_bool']
booking_bool = data['booking_bool']

def find_number_of_nans(boolean_data):
    nans = 0
    for i in boolean_data:
        if i == np.nan:
            print("nan")
            nans+=1
        if i is None:
            print("null")
            nans+=1
    print(nans)
    return nans

find_number_of_nans(srch_saturday_night_bool)
find_number_of_nans(random_bool)
find_number_of_nans(click_bool)
find_number_of_nans(booking_bool)


