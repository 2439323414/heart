# 作者：zgx
# 时间：2021/10/26 15:59
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

# 查看不同特征之间的相关关系
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(data=corr, annot=True, square=True, fmt='.2f')
plt.show()

# 心脏病和cp（心绞痛类型）之间的关系
# cpDf = df[['cp', 'target']]
# sns.countplot(data=cpDf, x='cp', hue='target', palette='Set2')
# plt.xlabel('胸痛类型（0=典型心绞痛；1=非典型心绞痛；2=非心绞痛；3=没有症状）', fontsize=12)
# plt.ylabel('人数', fontsize=12)
# plt.show()

# 心率和年龄也有一定的关系，可以结合考察心脏病，心率，年龄
# thaDf = df[['thalach', 'target']]
# thaDf['age_range'] = pd.cut(df['age'], bins=[0, 18, 40, 60, 100],
#                             labels=['儿童', '青年', '中年', '老年'],
#                             include_lowest=True, right=False)
# sns.swarmplot(data=thaDf, x='age_range', y='thalach', hue='target')
# plt.xlabel('年龄段', fontsize=12)
# plt.ylabel('最大心率', fontsize=12)
# plt.show()

# 静息血压和心脏病的关系
# trebpsDf = df[['trestbps', 'target']]
# sns.boxplot(trebpsDf['target'], trebpsDf['trestbps'])
# plt.xlabel('是否患心脏病（0=否，1=是）', fontsize=12)
# plt.ylabel('人数', fontsize=12)
# plt.title('心脏病与静息血压的关系')
# plt.show()
