import numpy as np
import scipy
import matplotlib.pyplot as plt
from numpy.random import normal, randint, uniform
from sklearn.preprocessing import StandardScaler, MinMaxScaler

## PARAM ONLY FOR DEBUG PURPOSE
DEBUG_PRINT = 0
CAL_DEBUG_OPT = 0
VAL_DEBUG_PRINT = 1

## Set Params HERE
#feat_num = 17
num_sets = 5000
iters = 10
inner_iters = 10
#max_val = 1
#val_offset = 0.1

cum_acc = 0

for iter in range(iters):
    for feats in range(inner_iters):
        np.random.seed(seed = iter * feats)
        max_val = randint(1, 100)
        val_offset = randint(1, 100)
        feat_num = feats+ 5

        ## Generate Dataset for Regressor
        #X = normal(0, 1, size = (num_sets, feat_num))
        X = uniform(size = (num_sets, feat_num)) * max_val - val_offset

        #sc = StandardScaler()
        #X_norm = sc.fit_transform(X)
        mms = MinMaxScaler()
        X_norm = mms.fit_transform(X)
        #weight = np.linspace(1, 300, feat_num)
        weight = randint(1, 100, size = feat_num)
        weight = np.floor(weight)
        y = np.dot(X_norm, weight)

        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import LinearSVR
        from sklearn.neural_network import MLPRegressor
        from sklearn.linear_model import LinearRegression

    ########################################################
    ########################################################
        #lreg = LinearRegression()
        #lreg.fit(X_norm, y)

        #svr = SVR()
        #svr.fit(X_norm, y)

        #lsvr = LinearSVR()
        #lsvr.fit(X_norm, y)

        #knr = KNeighborsRegressor(n_neighbors = 2)
        #knr.fit(X_norm, y)

        mlpr = MLPRegressor()
        mlpr.fit(X_norm, y)

    ########################################################
        model = mlpr
    ########################################################

        from miv_dimention_reduction import MIV
        miv = MIV(Model = model, threshold = 0.9, zeta = 0.1, score_ = None, is_clf = False)
        miv.fit(X_norm, y)
        X_selected = miv.transform(X_norm, y)

        if CAL_DEBUG_OPT:

            ex_weight = 0.2 * weight

            IV_cal = np.zeros(miv.IV.shape)

            for i in range(feat_num):
                IV_cal[:, i] = X_norm[:, i] * ex_weight[i]

            plt.show()

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

        order_feat = np.argsort(weight)[::-1]
        order_evr = np.array(list(miv.explained_MIV.keys()))
        acc = sum(order_feat == order_evr) / len(order_feat)

        if VAL_DEBUG_PRINT:
            for key, val in miv.explained_MIV.items():
                print("%2dth explained ratio : %.4f" %(key, val))

            print(order_feat)
            print(order_evr)

            print("ACCURACY : %.4f\n" %acc)

        cum_acc += acc
        print(miv.selected)

        from matplotlib import pyplot as plt
        fig = plt.figure(iter)
        plt.plot(order_feat, order_evr, 'bo');
        plt.xlabel("Feature Index of Weight")
        plt.ylabel("Feature Index from MIV")
        tmp_str = type(model).__name__
        plt.title("Check Feature Index Linearity, " + tmp_str)
        plt.show()
        fig.savefig("plot/%s_Val_Plot_%02d_%02d_%02d.png" %(tmp_str, max_val, val_offset, iter + 1), dpi = 300)
        plt.close()

fin_acc = cum_acc / (iters * feats)
print("FINAL ACCURACY OF THIS SIMULATION : %.4f\n" %fin_acc)

