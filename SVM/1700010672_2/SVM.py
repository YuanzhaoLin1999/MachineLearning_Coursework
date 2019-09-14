#SVM
from sklearn import svm
import numpy as np
import pandas as pd

def loadData(filename):
    """默认csv文件第一行是列属性，以下每行是训练数据
    并且数据的最后一列必然是数据的label"""
    frame = pd.read_csv(filename)#读为DataFrame格式
    labels = frame['label']#获取标记
    attrset = list(frame.columns)
    attrset.remove('label')
    dataset = list(frame.values)
    for i in range(frame.shape[0]):#遍历所有数据
        dataset[i] = list(dataset[i])
        dataset[i].pop()
    return dataset, list(labels), attrset #返回数据集(Dataset)和属性集(list)

dataset, labels, attrset = loadData('data3.0alpha.csv')
kernel = ('linear', 'rbf')
clf1 = svm.SVC(C=1, kernel='linear')
clf2 = svm.SVC(C=1, kernel='rbf')

ans = (clf1.fit(dataset, labels), clf2.fit(dataset, labels))

for i in range(2):
    svs = ans[i].support_vectors_
    print('核函数为',kernel[i],'的支持向量为：')
    for j in range(len(svs)):
        print(svs[j])
    print('支持向量的个数为',len(svs))
print(clf2.predict(dataset))
# clf3=svm.SVC(kernel='rbf',gamma='scale')
# clf3.fit(dataset,labels)
# a=clf3.predict(dataset)
# print(a)
# print(len(clf3.support_vectors_))