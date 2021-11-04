# 作者：zgx
# 时间：2021/10/21 10:48
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 解决matplotlib中文问题
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# 导入数据
df = pd.read_csv('data/heart.csv')

df.info()

df.describe()

first = pd.get_dummies(df['cp'], prefix="cp")
second = pd.get_dummies(df['slope'], prefix="slope")
thrid = pd.get_dummies(df['thal'], prefix="thal")

df = pd.concat([df, first, second, thrid], axis=1)
df = df.drop(columns=['cp', 'slope', 'thal'])
df.head(3)

y = df.target.values
X = df.drop(['target'], axis=1)

print(X.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6)  # 随机种子6

from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train = standardScaler.transform(X_train)
X_test = standardScaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                               metric_params=None, n_jobs=1, n_neighbors=15, p=3,
                               weights='distance')
# 无参数 训练集86 测试集81
# knn_clf = KNeighborsClassifier()

knn_clf.fit(X_train, y_train)

score = knn_clf.score(X_train, y_train)
print(score)

score = knn_clf.score(X_test, y_test)
print(score)
