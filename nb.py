
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:48:51 2018

@author: jasap
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from pandas.api.types import is_numeric_dtype
data = pd.read_csv("drug.csv")

def learn(data,label = None,alpha = 0.01):
    if(type(label) != str):
        label = data.columns[-1]
    apriori = data[label].value_counts()
    apriori += alpha
    apriori = apriori / (apriori.sum())
    apriori = np.log(apriori)
    model = {}
    model['apriori'] = apriori
    for attribute in data.drop(label,axis = 1).columns:
        if is_numeric_dtype(data[attribute]):
            me = data.groupby([label])[attribute].mean()
            st = data.groupby([label]).std()[attribute]
            model[attribute] = [me,st]
            continue
        freq_matrix = pd.crosstab(data[attribute],data[label])
        freq_matrix += alpha
        cont_matrix = freq_matrix / (freq_matrix.sum())# + alpha * len(data[attribute]))
        model[attribute] = np.log(cont_matrix)
    return model

new = pd.read_csv("C:\\Users\\jasap\\Downloads\\RAMU\\novi.csv")

print(learn(data))

def predict(model, new):
    prediction = {}
    for label_class in model['apriori'].index:
        probability = model['apriori'][label_class]
        for attr in new.columns:
            if is_numeric_dtype(data[attr]):#PAAAAAZNJA OPVO SE LOGARITMUJE I NIKE *= nego +=
                probability +=np.log( norm.pdf(new[attr],model[attr][0][label_class],model[attr][1][label_class]))
                continue
            probability += model[attr][label_class][new[attr][0]]
            prediction[label_class] = np.exp(probability)
    z = sum(prediction.values())
    for k in prediction.keys():
        prediction[k] = prediction[k] / z
    return prediction

print(predict(learn(data),new))

