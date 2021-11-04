简介：
数据集文件为data下的heart.csv
logisticRegression.py包含逻辑回归模型，以及整合了决策树、随机森林、kNN
DecisionTree.py 决策树模型
RandomForest.py 随机森林模型
KNN.py KNN模型
draw.py  一些画图算法


数据集描述:
age 年龄
sex 性别 1=male,0=female
cp  胸痛类型(4种) 值1:典型心绞痛，值2:非典型心绞痛，值3:非心绞痛，值4:无症状
trestbps 静息血压 
chol 血清胆固醇
fbs 空腹血糖 >120mg/dl ,1=true; 0=false
restecg 静息心电图(值0,1,2)
thalach 达到的最大心率
exang 运动诱发的心绞痛(1=yes;0=no)
oldpeak 相对于休息的运动引起的ST值(ST值与心电图上的位置有关)
slope 运动高峰ST段的坡度 Value 1: upsloping向上倾斜, Value 2: flat持平, Value 3: downsloping向下倾斜
ca  The number of major vessels(血管) (0-3)
thal A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
       一种叫做地中海贫血的血液疾病(3 =正常;6 =固定缺陷;7 =可逆转缺陷)
target 生病没有(0=no,1=yes)


