import numpy as np 
import matplotlib.pyplot as plt
from time import time
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import load_boston #波士顿房价数据集，用于回归任务
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.datasets import load_diabetes#糖尿病数据集，用于回归任务
t = time()
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
depth = 2
n = 100

reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=depth),
                            n_estimators= n)
#reg = BaggingRegressor(DecisionTreeRegressor(max_depth=depth),n_estimators=n)
#reg = BaggingRegressor(AdaBoostRegressor(DecisionTreeRegressor(max_depth=depth),
#                            n_estimators= n), n_estimators= n)
#reg = AdaBoostRegressor(BaggingRegressor(DecisionTreeRegressor(max_depth=depth),
#                           n_estimators=n), n_estimators= n)

score = cross_validate(reg, X, y,cv =10,scoring= 'r2',return_train_score=True)
for x in score:
    print(x,":",score[x])
    print('average',x,":",np.mean(score[x]))