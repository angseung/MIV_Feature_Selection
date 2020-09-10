# 필요한 모듈 import
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 당뇨병 데이터셋 가져오기
diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data[:, np.newaxis, 2]

# 배열 크기
diabetes_X.shape

# 훈련 세트, 테스트 세트 데이터 분리
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# 훈련 세트, 테스트 세트 타겟값 분리
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# 선형회귀 모델 생성
regr = LinearRegression()

# 훈련 세트를 사용해서 모델 훈련
regr.fit(diabetes_X_train, diabetes_y_train)

# scikit-learn은 훈련데이터에서 유도된 속성은 항상 끝에 밑줄을 붙입니다.
# regr.coef_, regr.intercept_ 처럼 밑줄을 붙임으로써 사용자가 지정한 변수와 구분할 수 있습니다.

# 계수(가중치)
print('Coefficients: ', regr.coef_)

# 편향(절편)
print('Intercept: ', regr.intercept_)

# 평균 제곱근 편차
print("Mean squared error: %.2f" % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))

# 훈련데이터 성능
print('TrainSet score: %.2f' % regr.score(diabetes_X_train, diabetes_y_train))

# 테스트데이터 성능
print('TestSet score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# 도표 결과
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()