# 作者：zgx
# 时间：2021/10/24 19:02
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
# 数据处理
# 对特征中非连续型数值(cp,slope,thal)特征进行处理
first = pd.get_dummies(df['cp'], prefix="cp")
second = pd.get_dummies(df['slope'], prefix="slope")
thrid = pd.get_dummies(df['thal'], prefix="thal")

df = pd.concat([df, first, second, thrid], axis=1)
df = df.drop(columns=['cp', 'slope', 'thal'])
df.head(3)
print(df.head())

y = df.target.values
X = df.drop(['target'], axis=1)

print(X.shape)
# 分割数据集,并进行标准化处理
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6)  # 随机种子6

from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train = standardScaler.transform(X_train)
X_test = standardScaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, random_state=666, oob_score=True, n_jobs=-1)
rf_clf.fit(X_train, y_train)

# score = log_reg.score(X_train, y_train)
# print(score)

score = rf_clf.score(X_test, y_test)
print(score)

y_probabilities_rf = rf_clf.predict_proba(X_test)[:, 1]

# 求面积,相当于求得分
from sklearn.metrics import roc_auc_score  # auc:area under curve

roc_score = roc_auc_score(y_test, y_probabilities_rf)
print(roc_score)
