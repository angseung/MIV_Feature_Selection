import numpy as np
import scipy
import matplotlib.pyplot as plt
from numpy.random import normal, randint, uniform
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

## PARAM ONLY FOR DEBUG PURPOSE
DEBUG_PRINT = 0
CAL_DEBUG_OPT = 0
VAL_DEBUG_PRINT = 1

## Set Params HERE
feat_num = 10
num_sets = 5000
iters = 1
#max_val = 1
#val_offset = 0.1

cum_acc = 0

for iter in range(iters):
    np.random.seed(seed = 10)

    from sklearn.datasets import load_diabetes
    X, y = load_diabetes(return_X_y = True)
    mms = MinMaxScaler()
    X_norm = mms.fit_transform(X)
    #X_norm = X

    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import LinearSVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import LinearRegression

########################################################
########################################################
    lreg = LinearRegression()
    lreg.fit(X_norm, y)

    #svr = SVR(max_iter = 2000, verbose = True)
    #svr.fit(X_norm, y)

    #lsvr = LinearSVR()
    #lsvr.fit(X_norm, y)

    #knr = KNeighborsRegressor(n_neighbors = 2)
    #knr.fit(X_norm, y)

    #mlpr = MLPRegressor(max_iter = 1000, verbose = True)
    #mlpr.fit(X_norm, y)

########################################################
    model = lreg
########################################################

    from miv_dimention_reduction import MIV
    miv = MIV(Model = model, threshold = 0.9, zeta = 0.1, score_ = None, is_clf = False)
    miv.fit(X_norm, y)
    X_selected = miv.transform(X_norm, y)

    # mlpr2 = MLPRegressor(max_iter = 2000, verbose = True)
    # mlpr.fit(X_norm[:, miv.selected], y)
    lreg2 = LinearRegression()
    lreg2.fit(X_selected, y)
    print(lreg.score(X_norm, y), lreg2.score(X_selected, y))
    mse1 = mean_squared_error(lreg.predict(X_norm), y)
    mse2 = mean_squared_error(lreg2.predict(X_selected), y)
    print("%.4f, %.4f \n" %(mse1, mse2))
