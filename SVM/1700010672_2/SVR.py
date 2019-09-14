from sklearn import svm
import numpy as np
import pandas as pd

frame = pd.read_csv('data3.0alpha.csv')
labels = frame['label']
X = frame['密度']
Xm = []
for x in X:
    Xm.append([x])
y = frame['含糖率']
clf = svm.SVR(gamma='scale',kernel='rbf')
clf.fit(Xm, y)