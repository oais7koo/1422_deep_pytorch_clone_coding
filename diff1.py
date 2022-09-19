from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split



name = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv('psdata/ps1020_iris/iris.data', names=name)

# 훈련과 테스트 데이터셋 분리
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
s = StandardScaler()  # 특성 스케일링, 평균이 0, 표준편차가 1이되도록 반환
x_train = s.fit_transform(x_train)
x_test = s.fit_transform(x_test)

#knn = kNeighborsClassifier(n_neighbors=50)
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(x_train, y_train)
