import pandas as pd
import numpy as np 
from scipy.optimize import linear_sum_assignment
from itertools import accumulate 
from bisect import bisect_left
import matplotlib.pyplot as plt

FILE_PATH = ""
START_IDX_DATA = 1
NUM_PER_MIG = [10, 14, 10, 10, 7, 11]
ACC_NUM_PER_MIG = list(accumulate(NUM_PER_MIG))

@np.vectorize
def transform_cost(x):
    if x < 3:
        return x
    return x + 20
    
@np.vectorize
def reduce_repeated(x):
    return bisect_left(ACC_NUM_PER_MIG, x + 1)

df = pd.read_csv(FILE_PATH).sample(frac=1)
transformed_rankings = transform_cost(df.iloc[:, START_IDX_DATA:].to_numpy())
cost_matrix = np.repeat(transformed_rankings, NUM_PER_MIG, axis=1)
_, col_idx = linear_sum_assignment(cost_matrix)
assignment = reduce_repeated(col_idx)
df["Assignment"] = assignment + 1
df.sort_index()

print(df.head())
plt.hist([df.iloc[i, df.loc[i, "Assignment"]] for i in range(len(df))])
plt.show()



