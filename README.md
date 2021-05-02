# HSIC regularized Kernel Ridge Regression

This repo contains the code to perform fair kernel learning.

The model is based on [Fair Kernel Learning](https://arxiv.org/abs/1710.05578) and [Kernel Dependence Regularizers and Gaussian Processes with Applications to Algorithmic Fairness](https://arxiv.org/abs/1911.04322).
The code is modified from repo [fairgp_code](https://github.com/Mick116/fairgp_code).

To reproduce the results, please read data and parameters that we provided from files, instead of generate new parameters.

Step 1: Run ` pip install -r requirements.txt` to set up environment.

Step 2: Run `fair_KRR.py`. **Attention: some of the code is used to generating parameters for grid search, but if you run the file directly, it will overwrite all the generated parameters by reading parameters from files to reproduce result from the report.**

To use approximation of HSIC to speed up the calculation, set `auto=True` in `HSIC, pvalue = Hsic().test(s_train, y_train_pred, workers=-1, auto=True)`(from `perform_kernel_regression_fair_learning.py`, function `fair_regression`).

`data_preprocessing/py`: data pre-processing, modified from repo [Communities-Crime
](https://github.com/vbordalo/Communities-Crime)

`visual`: contains file to visualize result.