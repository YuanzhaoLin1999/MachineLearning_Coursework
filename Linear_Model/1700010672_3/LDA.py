#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import datasets
from matplotlib import colors

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
    return dataset, list(labels), attrset


X, y, labels = loadData('data3.0alpha.csv')
clf = LinearDiscriminantAnalysis()

fig, ax = plt.figure()
Xp, Xn = X[y == 1], X[y == -1]
print(Xp, Xn)
plt.show()