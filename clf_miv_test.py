from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='liblinear', multi_class='auto', C=100.0, random_state=1)
lr.fit(X_train_std, y_train)

from miv_200324 import miv, miv_clf

[selected, cum_MIV, IV, MIV, explained_MIV, explained_pca] = \
    miv_clf(clf = lr, X = X_train, zeta = 0.1, threshold = 0.9, is_clf = True)