# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 12:47:02 2019

@author: Hemant Shinde
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df_train=pd.read_csv('../train.csv')
#find useless coloumn-- if non Null>1200 then keep else delete
#find categorical features and non categorical features
#delete categorical features whose unique values >8
#find non-cat columns using mean/mode/median
#fillna cat columns using mode/median
#create dummy columns from cat-columns
#create final data frame for train
#create random forest/xgboost
#create data frame as per submission file and submit to kaggal


