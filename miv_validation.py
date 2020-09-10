from sklearn.linear_model import LinearRegression
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numpy.random import normal, randint, uniform
from sklearn.preprocessing import StandardScaler, MinMaxScaler

## Set Params HERE
feat_num = 14
num_sets = 20000
iters = 1

for iter in range(iters):
    np.random.seed(seed = iter)

    ## Generate Dataset for Regressor
    #X = normal(0, 1, size = (num_sets, feat_num))
    X = uniform(size = (num_sets, feat_num)) * 100 - 50
    #sc = StandardScaler()
    #X_norm = sc.fit_transform(X)
    mms = MinMaxScaler()
    X_norm = mms.fit_transform(X)
    #weight = np.linspace(1, 300, feat_num)
    weight = randint(1, 300, size = feat_num)
    weight = np.floor(weight)
    y = np.dot(X_norm, weight)

    # Train Linear Regressor
    lr = LinearRegression()
    lr.fit(X_norm, y)

    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import LinearSVR
    from sklearn.neural_network import MLPRegressor
    #svr = SVR()
    #knr = KNeighborsRegressor(n_neighbors = 2)
    #svr.fit(X_norm, y)
    #knr.fit(X_norm, y)
    mlpr = MLPRegressor()
    mlpr.fit(X_norm, y)
    #knr = KNeighborsRegressor(n_neighbors = 2)
    #svr.fit(X_norm, y)
    #knr.fit(X_norm, y)

    #lsvr = LinearSVR()
    #lsvr.fit(X_norm, y)

    model = mlpr
    from miv_dimention_reduction import MIV
    miv = MIV(Model = model, threshold = 0.9, zeta = 0.1, score_ = None, is_clf = False)
    miv.fit(X_norm, y)
    X_selected = miv.transform(X_norm, y)

    ex_weight = 0.2 * weight

    IV_cal = np.zeros(miv.IV.shape)

    for i in range(feat_num):
        IV_cal[:, i] = X_norm[:, i] * ex_weight[i]

    plt.show()

    DEBUG_PRINT = 0
    if DEBUG_PRINT:
        fig = plt.figure(1)

        for i in range(feat_num):
            plt.hist(X_norm[:, i])

        print(weight)
        print(weight[miv.selected])

        # Check Impact Value Linearity
        fig = plt.figure(2)
        plt.plot(IV_cal.flatten(), miv.IV.flatten(), 'ro')
        plt.xlabel("Calculated IV")
        plt.ylabel("IV from MIV")
        plt.title("Test Cal Linearity")
        plt.show()

        # Check Regression Linearity
        fig = plt.figure(3)
        plt.plot(lr.predict(X_norm), y, 'bo')
        plt.xlabel("predicted Y")
        plt.ylabel("Real Y")
        plt.title("Test Y Linearity")
        plt.show()

        for i in range(feat_num):
            print(" %dTH [mean = %.2f, var = %.2f]" %(i, X[:, i].mean(), X[:, i].std()))

    for key, val in miv.explained_MIV.items():
        print("%2dth explained ratio : %.4f" %(key, val))
    order_feat = np.argsort(weight)[::-1]
    print(order_feat)