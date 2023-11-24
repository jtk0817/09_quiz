import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from pandas.plotting import scatter_matrix

# 데이터 파일 경로
filename = "./data/09_irisdata.csv"

# 컬럼명 정의
column_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

# 데이터 불러오기
data = pd.read_csv(filename, names=column_names)
# 데이터 셋의 행렬 크기(shape)
print("데이터 셋의 행렬 크기:", data.shape)

# 데이터 셋의 요약
print("데이터 셋의 요약:\n", data.describe())

# 데이터 셋의 클래스 종류
print("데이터 셋의 클래스 종류:\n", data.groupby('class').size())

# scatter_matrix 그래프 저장
scatter_matrix(data, alpha=0.2, figsize=(10, 10), diagonal='hist')
plt.savefig('scatter_matrix_plot.png')

# 독립 변수 X와 종속 변수 Y로 분할
X = data[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']]
Y = data['class']

# 모델 학습 및 평가
model = DecisionTreeClassifier()
kfold = KFold(n_splits=10, shuffle=True)

# Cross Validation을 사용하여 평가
results = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')

# K-fold의 평균 정확도 출력
print("K-fold의 평균 정확도:", results.mean())

