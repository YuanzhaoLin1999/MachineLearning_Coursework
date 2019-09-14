#!/usr/bin/env python
# coding: UTF-8
from math import log
import numpy as np 
import pandas as pd

#类：学习数据
class T_data:
    def __init__(self,data,label):
        """data must be a dictionary 特征 -> 数据"""
        self.data = data
        self.label = label

#类：数据集
class DataSet:
    def __init__(self):
        self.datas = []
        self.labels = dict()

    def add(self, data):
        """data must be T_data type"""
        self.datas.append(data)
        label = data.label
        if label not in self.labels:
            self.labels[label] = 0
        self.labels[label] += 1
    @staticmethod
    def is_numeric(data):
        return isinstance(data, int) or isinstance(data, float)

    def gini(self):
        counts = self.labels
        impurity = 1
        for l in counts:
            p = counts[l]/len(self.datas)
            impurity -= pow(p,2) 
        return impurity
    
    def gini_index(self, a):
        """calculate the gini_index using character a"""
        g_index = 0
        Dv = dict()
        for d in self.datas:
            a_info = d.data[a]
            if a_info in Dv:
                Dv[a_info].add(d)
            else:
                new_dataset = DataSet()
                new_dataset.add(d)
                Dv[a_info] = new_dataset
        for x in Dv:
            N = len(self.datas) #|D|
            Nv = len(Dv[x].datas)#|Dv|
            g_index += Dv[x].gini() * Nv / N
        return g_index, Dv#Dv是以特征a划分的，字典，key是a特征的各个具体取值

    def entropy(self):
        entropy = 0
        counts = self.labels
        for l in counts:
            p = counts[l] / len(self.datas)
            if p == 0:
                continue
            entropy -= p*log(p,2)
        return entropy
    
    def info_gain(self,a):
        """calculater the information gain of 
        character a"""
        entro = self.entropy()
        Dv = dict()
        for d in self.datas:
            a_info = d.data[a]
            if a_info in Dv:
                Dv[a_info].add(d)
            else:
                new_dataset = DataSet()
                new_dataset.add(d)
                Dv[a_info] = new_dataset
        for x in Dv:
            N = len(self.datas) #|D|
            Nv = len(Dv[x].datas)#|Dv|
            entro -= Dv[x].entropy() * Nv / N
        return entro, Dv
        

def loadData(filename):
    """默认csv文件第一行是列属性，以下每行是训练数据
    并且数据的最后一列必然是数据的label"""
    frame = pd.read_csv(filename)#读为DataFrame格式
    label = frame['label']#获取标记
    attrset = list(frame.columns)
    attrset.remove('label')
    f = frame.T#转置，以便获得Series对象，转化成dict类型
    dataset = DataSet()
    for i in range(frame.shape[0]):#遍历所有数据
        d = dict(f[i])
        del d['label']
        t = T_data(d, label[i])
        dataset.add(t)
    return dataset, attrset #返回数据集(Dataset)和属性集(list)
