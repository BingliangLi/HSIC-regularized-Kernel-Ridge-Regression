#%%
import GPy
import numpy as np
from hyppo.independence import Hsic
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import multiprocessing as mp
import pandas as pd

res_everything = pd.DataFrame(columns = ['sc_x', 'lmda', 'mu', 'rmse', 'hsic'])

# perform kernel regression fair learning
## the H function
def centre_mat(n_samples):
    mat_1 = np.ones(n_samples)
    # print mat_1
    # print np.outer(mat_1,mat_1)
    H = np.diag(mat_1) - (1.0 / n_samples) * np.outer(mat_1, mat_1)

    return H


# step 1: define the regression function
def fair_regression(x_train, y_train, x_test, y_test, s_train, s_test, sc_x=1.0, sc_s=1.0, lmda=0.1, mu=0.0):
    """
    This is the fair regression function that outputs the prediction, rmse and hsic
    x_train,y_train,x_test,y_test: training and testing samples
    s_train,s_test: sensitive variable
    sc_x: lengthscale for kernel on x
    sc_s: lengthscale for kernel on s
    lmda: penalty parameter for function norm
    mu: penalty parameter for HSIC
    """
    # print('inside fair_regression')
    # the dimensions of training set
    stime = time.time()
    input_number = x_train.shape[0]
    input_dims = x_train.shape[1]
    sensitive_dims = s_train.shape[1]

    # the centering matrix
    H = centre_mat(input_number)

    # specify the kernel
    k_x = GPy.kern.RBF(input_dims, lengthscale=sc_x)
    k_s = GPy.kern.RBF(sensitive_dims, lengthscale=sc_s)

    # compute Gram matrix
    k_xx = k_x.K(x_train, x_train)
    k_ss = k_s.K(s_train, s_train)

    # k_ssh = np.matmul(k_ss, H)
    hk_ssh = H.dot(k_ss).dot(H)

    # compute the prediction
    k_test = k_x.K(x_test, x_train)
    inside_inv = k_xx + input_number * lmda * np.eye(input_number) + (mu / input_number) * np.dot(hk_ssh, k_xx)
    est_beta = np.linalg.solve(inside_inv, y_train)
    y_pred = np.dot(k_test, est_beta)
    # compute error
    error_pred = np.sqrt(np.sum((y_pred - y_test) ** 2) / y_test.shape[0])

    # measure unfairness
    y_train_pred = np.dot(k_xx, est_beta)
    # k_yy = np.outer(y_train_pred, y_train_pred)
    # HSIC = np.trace(np.dot(k_yy, hk_ssh)) / input_number ** 2

    HSIC, pvalue = Hsic().test(s_train, y_train_pred, workers=-1, auto=True)
    etime = time.time()
    print('error_pred: ', error_pred, 'HSIC: ', HSIC, 'time: ', (stime-etime))

    # Store everything: analytic propose only
    global res_everything
    res_everything = res_everything.append({'sc_x': sc_x, 'lmda': lmda, 'mu': mu, 'rmse': error_pred, 'hsic': HSIC}, ignore_index = True)


    return y_pred, error_pred, HSIC


# step 2: define cross validation function
def cross_v(x, y, s, par, mu, NumFolds):
    """
    x,y,s: the dataset
    par: the parameters for kernel and cross-validation; order: (par for x kernel, par for s kernel, penalization)
    NumFolds: number of cross validation
    """
    xs = np.concatenate((x, s), axis=1)

    sc_x = par[0]
    sc_s = par[1]
    lmda = par[2]

    err_mat = np.zeros(NumFolds)

    for ii in np.arange(NumFolds):
        xs_train, xs_test, y_train, y_test = train_test_split(xs, y, test_size=0.2)
        s_train = xs_train[:, -1].reshape(xs_train.shape[0], 1)
        s_test = xs_test[:, -1].reshape(xs_test.shape[0], 1)
        x_train = np.delete(xs_train, -1, 1)
        x_test = np.delete(xs_test, -1, 1)

        _, err_mat[ii], _ = fair_regression(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
                                            , s_train=s_train, s_test=s_test, sc_x=sc_x, sc_s=sc_s, lmda=lmda, mu=mu)

    err = np.mean(err_mat)

    return err, err_mat


# step 3: combine the pipeline
def FairLearning(x_train, y_train, x_test, y_test, s_train, s_test, par_list, mu, NumFolds):
    cv_length = len(par_list)
    cv_mat = np.zeros(cv_length)

    for ii in tqdm(np.arange(cv_length)):
        # print(ii)
        par = par_list[ii]

        cv_mat[ii], _ = cross_v(x=x_train, y=y_train, s=s_train, par=par, mu=mu, NumFolds=NumFolds)

    par_cv = par_list[np.argmin(cv_mat)]
    sc_x = par_cv[0]
    sc_s = par_cv[1]
    lmda = par_cv[2]

    y_pred, rmse, hsic = fair_regression(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                         s_train=s_train, s_test=s_test, sc_x=sc_x, sc_s=sc_s, lmda=lmda, mu=mu)

    results_list = [mu, y_pred, rmse, hsic, lmda, sc_x]
    print(results_list, '\n')
    return results_list


def fl_wrapper(args):
    x_train = args[0]
    y_train = args[1]
    x_test = args[2]
    y_test = args[3]
    s_train = args[4]
    s_test = args[5]
    par_list = args[6]
    mu = args[8]
    NumFolds = args[7]

    result = FairLearning(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                          s_train=s_train, s_test=s_test, par_list=par_list, mu=mu, NumFolds=NumFolds)
    return result


# define parallel function
def FairLearning_process(processes, x_train, y_train, x_test, y_test, s_train, s_test, par_list, mu_list, NumFolds):
    pool = mp.Pool(processes=processes)
    arg_list1 = [x_train, y_train, x_test, y_test, s_train, s_test, par_list, NumFolds]
    arg_list = []
    for mu in mu_list:
        arg_list2 = arg_list1 + [mu]
        arg_list.append(arg_list2)

    results = pool.map(fl_wrapper, arg_list)
    return results
