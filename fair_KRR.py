import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import multiprocess as mp
import itertools
import pickle
import time
# import gpflow
# import tensorflow as tf
import GPy
# from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import preprocessing
from tqdm import tqdm

import perform_kernel_regression_fair_learing
from perform_kernel_regression_fair_learing import FairLearning
from perform_kernel_regression_fair_learing import fl_wrapper
from perform_kernel_regression_fair_learing import FairLearning_process
from perform_kernel_regression_fair_learing import centre_mat
from perform_kernel_regression_fair_learing import cross_v

from hyppo.independence import Hsic

# for median_heuristic
from numpy.random import permutation
from scipy.spatial.distance import squareform, pdist, cdist

import time

# %%
X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv')

X = X.values.reshape(X.shape)[:, 1:]
y = y.values.reshape(y.shape)[:, 1:]
# %%
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#%%
def save_csv(array_name, filename):
    pd.DataFrame(array_name).to_csv(filename + '.csv')

save_csv(x_train, 'x_train')
save_csv(x_test, 'x_test')
save_csv(y_train, 'y_train')
save_csv(y_test, 'y_test')

#%%
x_train = pd.read_csv('x_train.csv').values[:, 1:]
x_test = pd.read_csv('x_test.csv').values[:, 1:]
y_train = pd.read_csv('y_train.csv').values[:, 1:]
y_test = pd.read_csv('y_test.csv').values[:, 1:]
#%%
s_train = x_train[:, 0].reshape(x_train.shape[0], 1)
s_test = x_test[:, 0].reshape(x_test.shape[0], 1)
# %%
# TODO Set parameters for grid search
# cross validation searching grid
def get_sigma_median_heuristic(X):
    n = X.shape[0]
    if n > 1000:
        X = X[permutation(n)[:1000], :]
    dists = squareform(pdist(X, 'euclidean'))
    median_dist = np.median(dists[dists > 0])
    sigma = median_dist / np.sqrt(2.)
    return sigma


median_dis = get_sigma_median_heuristic(x_train)
generated_sc_x_array = np.random.uniform(np.exp(-15), 1, 10)
heuristic_sc_x_array = np.random.randn(10, ) + median_dis
heuristic_sc_x_array = heuristic_sc_x_array[heuristic_sc_x_array > 0]
sc_x_array = np.concatenate((generated_sc_x_array, heuristic_sc_x_array, np.array(median_dis).reshape(1, )))

sc_s_array = np.array([0.5])  # np.random.uniform(0,5,25)

sc_x_array.sort()

# remove necessary sc_x(e.g., two sc_x that are two close)
temp_sc_x_array = list()
jump = False
beacon = sc_x_array[0]
temp_sc_x_array.append(beacon)
for i in range(len(sc_x_array)):
    if sc_x_array[i] - beacon >= 0.05:
        temp_sc_x_array.append(sc_x_array[i])
        beacon = sc_x_array[i]

sc_x_array = np.array(temp_sc_x_array)

lmda_array = (np.exp(-20 + np.arange(10)))

par_list = list(itertools.product(sc_x_array, sc_s_array, lmda_array))
#%%
try:
    with open('par_list_FKL', 'rb') as pf:
        par_list = pickle.load(pf)
except:
    pass

# %%
# define parallel function
def FairLearning_process(processes, x_train, y_train, x_test, y_test, s_train, s_test, par_list, mu_list, NumFolds):
    # pool = mp.Pool(processes=processes)
    # pool = Pool(processes=processes)
    arg_list1 = [x_train, y_train, x_test, y_test, s_train, s_test, par_list, NumFolds]
    arg_list = []
    for mu in mu_list:
        arg_list2 = arg_list1 + [mu]
        arg_list.append(arg_list2)

    # results = pool.map(fl_wrapper, arg_list)
    with mp.Pool(processes) as pool:
        # results = pool.map(fl_wrapper, arg_list)
        results = pool.map(fl_wrapper, arg_list)
    return results


# %%
par_list_t = list(itertools.product(sc_x_array, sc_s_array, lmda_array))

# y_pred, error_pred, HSIC = fair_regression(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
#                                    s_train=s_train, s_test=s_test)

pd.DataFrame(par_list_t).to_csv("par_list_t.csv")
# %%
processes = 6
NumFolds = 5
mu_list = [0, 0.1, 0.7, 2.0, 5.0, 10.0]

stime = time.time()
para_result = FairLearning_process(processes=processes, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                   s_train=s_train, s_test=s_test, par_list=par_list, mu_list=mu_list,
                                   NumFolds=NumFolds)

print("Time for FKRR CV search job: %.3f" % (time.time() - stime))
time = time.time() - stime

data = [x_train, y_train, x_test, y_test]
all = [data, para_result]

# %%
pickle_out1 = open("krr_result_gpy3", "wb")
pickle.dump(all, pickle_out1)
pickle_out1.close()

# %%
import pickle

with open('test', 'rb') as f:
    all = pickle.load(f)

'''
data: all[0][0:4] data = [x_train, y_train, x_test, y_test]

para_result: all[1][0:6] para_result = results = pool.map(fl_wrapper, arg_list)
fl_wrapper = results_list = [mu, y_pred, rmse, hsic, lmda, sc_x]

mu: all[1][n][0]
y_pred: all[1][n][1]
rmse: all[1][n][2]
hsic: all[1][n][3]
lmda: all[1][n][4]
sc_x: all[1][n][5]
'''
