import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets 
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import LeaveOneOut


datas = [datasets.load_iris(), datasets.load_wine()]
title = ['iris', 'wine']
#交叉验证的函数实现起来也并不复杂，这里直接调用现成的包
for i in range(2):
    X, y = datas[i].data, datas[i].target
    clf = LogisticRegressionCV(cv = LeaveOneOut(), random_state=0,max_iter=1000,
                               solver = 'lbfgs',multi_class = 'multinomial')
    clf.fit(X, y)
    print(clf.score(X,y))