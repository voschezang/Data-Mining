import numpy as np
import util.data
import pandas as pd
np.random.seed(123)

data = pd.read_csv('data/training_set_VU_DM_clean.csv', sep=';')

# CF matrix
# TODO use clusters
scores = util.data.scores_df(data)

print("Saving file")
scores.to_csv('data/scores_train.csv', sep=';', index=False)
