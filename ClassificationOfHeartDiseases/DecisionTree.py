# 作者：zgx
# 时间：2021/10/21 11:06
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

print(X)
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6)  # 随机种子6

from sklearn.preprocessing import StandardScaler



standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train = standardScaler.transform(X_train)
X_test = standardScaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier


# 无参数调优 72 参数调优后为78
dt_clf = DecisionTreeClassifier(splitter='random', random_state=7, max_depth=10, max_leaf_nodes=40,criterion='gini')

dt_clf.fit(X_train, y_train)

score = dt_clf.score(X_train, y_train)
print(score)

score = dt_clf.score(X_test, y_test)
print(score)
