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
# 数据处理
# 对特征中非连续型数值(cp,slope,restecg,thal)特征进行处理
first = pd.get_dummies(df['cp'], prefix="cp")
second = pd.get_dummies(df['slope'], prefix="slope")
thrid = pd.get_dummies(df['thal'], prefix="thal")
# fourth =  pd.get_dummies(df['thalach'],prefix='thalach')


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

from sklearn.linear_model import LogisticRegression

# 'C': 0.01, 'class_weight': 'balanced', 'penalty': 'l2'

log_reg = LogisticRegression(C=1.0, class_weight='balanced', dual=False, fit_intercept=True,
                             intercept_scaling=1, max_iter=500, multi_class='ovr', n_jobs=1,
                             penalty='l2', random_state=None, solver='saga', tol=0.0001,
                             verbose=0, warm_start=False)

log_reg.fit(X_train, y_train)

# score = log_reg.score(X_train, y_train)
# print(score)

score = log_reg.score(X_test, y_test)
print("逻辑回归：", score)

# # 使用网格搜索找出更好的模型参数
# param_grid = [
#     {
#         'C': [0.01, 0.1, 1, 10, 100],
#         'penalty': ['l2', 'l1'],
#         'class_weight': ['balanced', None]
#     }
# ]
#
# from sklearn.model_selection import GridSearchCV
#
# grid_search = GridSearchCV(log_reg, param_grid, cv=10, n_jobs=-1)
#
# grid_s = grid_search.fit(X_train, y_train)
# print(grid_s)
#
# log_reg = grid_search.best_estimator_
# log_train_score = log_reg.score(X_train, y_train)
# print(log_train_score)
#
# log_test_score = log_reg.score(X_test, y_test)
# print(log_test_score)

# 决策树
from sklearn.tree import DecisionTreeClassifier

# 无参数调优 72 参数调优后为78
dt_clf = DecisionTreeClassifier(splitter='random', random_state=7, max_depth=10, max_leaf_nodes=40)

dt_clf.fit(X_train, y_train)

# score = dt_clf.score(X_train, y_train)
# print(score)

score = dt_clf.score(X_test, y_test)
print("决策树：", score)

from sklearn.neighbors import KNeighborsClassifier

# knn_clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#                                metric_params=None, n_jobs=1, n_neighbors=15, p=3,
#                                weights='distance')
# 无参数 训练集86 测试集81
knn_clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1,
                               n_neighbors=15, p=3, weights='distance')

knn_clf.fit(X_train, y_train)

score = knn_clf.score(X_train, y_train)
print(score)

score = knn_clf.score(X_test, y_test)
print("KNN：", score)

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, random_state=666, oob_score=True, n_jobs=-1)
rf_clf.fit(X_train, y_train)

# score = log_reg.score(X_train, y_train)
# print(score)

score = rf_clf.score(X_test, y_test)
print("随机森林：", score)

# 绘制逻辑回归，KNN和决策树的混淆矩阵

from sklearn.metrics import confusion_matrix

i = 1
fig1 = plt.figure(figsize=(2 * 3, 1 * 4))

# estimator_dict = {'Logistic Regression': log_reg, 'KNN': knn_clf, 'Decision Tree': dt_clf, 'Random Forest': rf_clf}
# for key, estimator in estimator_dict.items():
#     # 绘制混淆矩阵
#     pred_y = estimator.predict(X_test)
#     matrix = pd.DataFrame(confusion_matrix(y_test, pred_y))
#     ax1 = fig1.add_subplot(1, 4, i)
#     sns.heatmap(matrix, annot=True, cmap='OrRd')
#     plt.ylabel('实际值0/1', fontsize=12)
#     plt.xlabel('预测值0/1', fontsize=12)
#     plt.title('Confusion Matrix -- %s ' % key)
#     i += 1
# plt.show()

pred_y = rf_clf.predict(X_test)
matrix = pd.DataFrame(confusion_matrix(y_test, pred_y))
ax1 = fig1.add_subplot(2, 2, 1)
sns.heatmap(matrix, annot=True, cmap='OrRd')
plt.ylabel('实际值0/1', fontsize=12)
plt.xlabel('预测值0/1', fontsize=12)
plt.title('Random Forest ')
plt.show()

