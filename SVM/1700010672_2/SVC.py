from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_meshgrid(x, y, i=1, h=.02): #创建窗口
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - i, x.max() + i
    y_min, y_max = y.min() - i, y.max() + i
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

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
clf = svm.SVC(kernel = 'rbf', gamma='scale',max_iter=1)
c = clf.fit(X, y)

frame = pd.read_csv('data3.0alpha.csv')
X0 = frame['密度']
X1 = frame['含糖率']
fig, ax = plt.subplots(1,1)
xx, yy = make_meshgrid(X0, X1, i=0.2)
plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('density',rotation = 30)
ax.set_ylabel('sugary', rotation= 30)
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('Kernel = rbf')
print(len(clf.support_vectors_))
plt.show()