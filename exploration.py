import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pickle
import util.plot
import util.data
import numpy as np
from pandas.plotting import scatter_matrix

np.random.seed(123)

data = pd.read_csv('data/training_set_VU_DM_clean.csv', sep=';',nrows=10000)

# some feature engineering
data["srch_person_per_room"] = (data["srch_adults_count"]+data["srch_children_count"])/data["srch_room_count"]
data.loc[~(data["srch_person_per_room"]<10000),"srch_person_per_room"] = 0
data["srch_adults_per_room"] = data["srch_adults_count"]/data["srch_room_count"]

columns = list(data.columns)
srch_crit  = [k for k in columns if 'srch' in k]
correlation_grid(data, srch_crit[2:8])

for k in columns[3:]:
    makeBarchart(data,k,len(set(data[k])))
    


scatter_matrix(data[srch_crit[2:8]], figsize=(6, 6))
plt.show()


def makeBarchart(data, k, n):
    class Encoders:
        discretizers = {}
        encoders = {}
    newData = pd.DataFrame(data[k])
    #util.data.discretize(newData, k, Encoders,n)
    newData["click_bool"] = data["click_bool"]
    newData["booking_bool"]= data["booking_bool"]
    # data to plot
    clicks = np.zeros(n)
    books = np.zeros(n)
    sets = list(set(newData[k]))
    for i in range(0,n):
        clicks[i] = np.mean(newData.loc[newData[k]==sets[i],"click_bool"])
        books[i] = np.mean(newData.loc[newData[k]==sets[i],"booking_bool"])
    
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n)
    bar_width = 0.35
    opacity = 0.8
    
    rects1 = plt.bar(index, clicks, bar_width,
    alpha=opacity,
    color='b',
    label='clicks')
    
    rects2 = plt.bar(index + bar_width, books, bar_width,
    alpha=opacity,
    color='g',
    label='books')
    
    plt.xlabel(k)
    plt.ylabel('mean')
    plt.title('mean by type')
    plt.xticks(index + bar_width, (sets))
    plt.legend()
    
    plt.tight_layout()
    plt.show()