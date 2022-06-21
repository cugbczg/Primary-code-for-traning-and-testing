import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

data_Set = []
data_train_x = []
data_Set_x = []
data_Set_y = []

fileIn = open(r"E:\\2\\新建结果6\\train.csv",encoding = "utf-8")
fileIn1 = open(r"E:\\2\\新建结果6\\test.csv",encoding = "utf-8")

#为了拆分数据集，不过有更简单的方法，用split相关
# for line in fileIn.readlines():
#    lineArr = line.strip().split(',')
#    data_Set.append(lineArr)
#    data_Set_x.append(lineArr[5:15])
#    data_Set_y.append(lineArr[1:2])
#
# for line1 in fileIn1.readlines():
#        lineArr1 = line1.strip().split(',')
#        data_Set.append(lineArr1)
#        data_Set_.append(lineArr1[4:14])

df = pd.read_csv(fileIn)
df = df.fillna(0)
X = df.values[:, 3:39]
features = df.columns[3:39]
Y = df.values[:, 1]  # 读取训练样本
y = Y.astype(np.int16)

df1 = pd.read_csv(fileIn1)
df1 = df1.fillna(0)
X1 = df1.values[:, 4:40]   # 读取预测样本
features1 = df1.columns[4:40]

for i in range(X.shape[0]):
    data_Set_x.append(X[i])
    data_Set_y.append(y[i])


data_train_x,data_test_x = train_test_split(data_Set_x,test_size = 0.3,random_state = 55)
data_train_y,data_test_y = train_test_split(data_Set_y,test_size = 0.3,random_state = 55)


clf1 = svm.SVC(C=1, kernel='rbf', probability=True, decision_function_shape="ovr", verbose=True, shrinking=False, max_iter=-1).fit(data_train_x, data_train_y)
# clf2 = svm.SVC(kernel='rbf').fit(data_train_x,data_train_y)
# clf3 = svm.SVC(kernel='poly').fit(data_train_x,data_train_y)
# clf4 = svm.SVC(kernel='sigmoid').fit(data_train_x,data_train_y)



data_test_result = clf1.predict(X1)
gailv = clf1.predict_proba(X1)

cz = pd.DataFrame(data_test_result)
cz.to_csv('E:\\2\\新建结果6\\yuce.csv')  # (要改)

cz1 = pd.DataFrame(gailv)
cz1.to_csv('E:\\2\\新建结果6\\gailv1.csv')  # (要改)

# print("linear线性核函数-训练集：",clf1.score(data_train_x, data_train_y))
# print("linear线性核函数-测试集：",clf1.score(data_test_x, data_test_y))
print("rbf径向基核函数-训练集：",clf1.score(data_train_x, data_train_y))
print("rbf径向基函数-测试集：",clf1.score(data_test_x, data_test_y))
# print("poly多项式核函数-训练集：",clf3.score(data_train_x, data_train_y))
# print("poly多项式核函数-测试集：",clf3.score(data_test_x, data_test_y))
# print("sigmoid神经元激活核函数-训练集：",clf4.score(data_train_x, data_train_y))
# print("sigmoid神经元激活核函数-测试集：",clf4.score(data_test_x, data_test_y))

# print('decision_function:\n', clf.decision_function(data_train_x))
# print('\npredict:\n', clf.predict(data_train_x))
