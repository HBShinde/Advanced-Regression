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
df_train.describe()
col_head=list(df_train.columns)
selected_col=[]
drop_col=[]
for col in df_train:
    if df_train[col].isna().sum()>=300:
        drop_col.append(col)
    else:
        selected_col.append(col)
selected_noncat=[]
selected_cat=[]
for col in selected_col:
    if df_train[col].dtype=='O':
        selected_cat.append(col)
    else:
        selected_noncat.append(col)
df_train[selected_noncat].describe()
y='SalePrice'
selected_col.remove(y)
selected_noncat.remove(y)
Nancount=[]
for col in selected_noncat:
    Nancount.append(df_train[col].isna().sum())
val=df_train['LotFrontage'].mean()
df_train['LotFrontage'].fillna(value=val,inplace=True)
val=df_train['MasVnrArea'].mode()[0]
df_train['MasVnrArea'].fillna(value=val,inplace=True)
df_train['GarageYrBlt'].fillna(value=df_train['GarageYrBlt'].median(),inplace=True)
df_train[selected_noncat +['SalePrice']].corr()['SalePrice']
ser=(df_train[selected_noncat +['SalePrice']].corr()['SalePrice'])
selected_noncat.remove('Id')
selected_noncat.remove('MSSubClass')
selected_noncat.remove('OverallCond')
selected_noncat.remove('BsmtFinSF2')
selected_noncat.remove('LowQualFinSF')
selected_noncat.remove('BsmtHalfBath')
selected_noncat.remove('BedroomAbvGr')
selected_noncat.remove('KitchenAbvGr')
selected_noncat.remove('EnclosedPorch')
selected_noncat.remove('3SsnPorch')
selected_noncat.remove('ScreenPorch')
selected_noncat.remove('PoolArea')
selected_noncat.remove('MiscVal')
selected_noncat.remove('MoSold')
selected_noncat.remove('YrSold')
#selected_noncat.remove('SalePrice')
#handling non categorical data


df_train['MasVnrType'].fillna(value=df_train['MasVnrType'].mode()[0],inplace=True)
df_train['BsmtQual'].fillna(value=df_train['BsmtQual'].mode()[0],inplace=True)
df_train['BsmtCond'].fillna(value=df_train['BsmtCond'].mode()[0],inplace=True)
df_train['BsmtExposure'].fillna(value=df_train['BsmtExposure'].mode()[0],inplace=True)
df_train['BsmtFinType1'].fillna(value=df_train['BsmtFinType1'].mode()[0],inplace=True)
df_train['BsmtFinType2'].fillna(value=df_train['BsmtFinType2'].mode()[0],inplace=True)
df_train['Electrical'].fillna(value=df_train['Electrical'].mode()[0],inplace=True)
df_train['GarageType'].fillna(value=df_train['GarageType'].mode()[0],inplace=True)
df_train['GarageFinish'].fillna(value=df_train['GarageFinish'].mode()[0],inplace=True)
df_train['GarageQual'].fillna(value=df_train['GarageQual'].mode()[0],inplace=True)
df_train['GarageCond'].fillna(value=df_train['GarageCond'].mode()[0],inplace=True)

selected_cat.remove('Neighborhood')
selected_cat.remove('Exterior1st')
selected_cat.remove('Exterior2nd')
selected_cat.remove('Condition1')
selected_cat.remove('Condition2')
selected_cat.remove('HouseStyle')
selected_cat.remove('RoofMatl')
selected_cat.remove('Functional')
selected_cat.remove('SaleType')

#one hot enncoing using getdummies
temp=selected_cat
df_temp=pd.get_dummies(df_train[temp],prefix=temp,drop_first=True)

df_final_train=pd.concat([df_train[selected_noncat],df_temp],axis=1)

X_train=df_final_train
Y_train=df_train['SalePrice']

#Randome forest model fit 

from sklearn.ensemble import RandomForestRegressor
RFC=RandomForestRegressor()

RFC.fit(X_train,Y_train)

RFC.score(X_train,Y_train)


#hadnling the test data
df_test=pd.read_csv('../test.csv')

df_test.info()
selected_col_test=[]
drop_col=[]
for col in df_test:
    if df_test[col].isna().sum()>=300:
        drop_col.append(col)
    else:
        selected_col_test.append(col)

#non cateh=gorical and categorical columns
non_cat_test=[]
cat_test=[]
for i in selected_col_test:
    if df_test[i].dtype=='O':
        cat_test.append(i);
    else:
        non_cat_test.append(i)
        
#handling NA data in test
df_test['LotFrontage'].fillna(value=df_train['LotFrontage'].mean(),inplace=True)
df_test['MasVnrArea'].fillna(value=df_train['MasVnrArea'].mean(),inplace=True)
df_test['BsmtFinSF1'].fillna(value=df_train['BsmtFinSF1'].mean(),inplace=True)
df_test['BsmtFinSF2'].fillna(value=df_train['BsmtFinSF2'].mean(),inplace=True)
df_test['BsmtUnfSF'].fillna(value=df_train['BsmtUnfSF'].mean(),inplace=True)
df_test['TotalBsmtSF'].fillna(value=df_train['TotalBsmtSF'].mean(),inplace=True)
df_test['BsmtFullBath'].fillna(value=df_train['BsmtFullBath'].mode()[0],inplace=True)
df_test['BsmtHalfBath'].fillna(value=df_train['BsmtHalfBath'].mode()[0],inplace=True)
df_test['GarageYrBlt'].fillna(value=df_train['GarageYrBlt'].mean(),inplace=True)
df_test['GarageCars'].fillna(value=df_train['GarageCars'].mean(),inplace=True)
df_test['GarageArea'].fillna(value=df_train['GarageArea'].mean(),inplace=True)




non_cat_test.remove('Id')
non_cat_test.remove('MSSubClass')
non_cat_test.remove('OverallCond')
non_cat_test.remove('BsmtFinSF2')
non_cat_test.remove('LowQualFinSF')
non_cat_test.remove('BsmtHalfBath')
non_cat_test.remove('BedroomAbvGr')
non_cat_test.remove('KitchenAbvGr')
non_cat_test.remove('EnclosedPorch')
non_cat_test.remove('3SsnPorch')
non_cat_test.remove('ScreenPorch')
non_cat_test.remove('PoolArea')
non_cat_test.remove('MiscVal')
non_cat_test.remove('MoSold')
non_cat_test.remove('YrSold')


#handling non categorical data

#cat_test.remove('Fence')
#cat_test.remove('FireplaceQu')

cat_test.remove('Neighborhood')
cat_test.remove('Exterior1st')
cat_test.remove('Exterior2nd')
cat_test.remove('Condition1')
cat_test.remove('Condition2')
cat_test.remove('HouseStyle')
cat_test.remove('RoofMatl')
cat_test.remove('Functional')
cat_test.remove('SaleType')

df_test['MSZoning'].fillna(value=df_train['MSZoning'].mode()[0],inplace=True)
df_test['Utilities'].fillna(value=df_train['Utilities'].mode()[0],inplace=True)
df_test['MasVnrType'].fillna(value=df_train['MasVnrType'].mode()[0],inplace=True)
df_test['BsmtCond'].fillna(value=df_train['BsmtCond'].mode()[0],inplace=True)
df_test['BsmtQual'].fillna(value=df_train['BsmtQual'].mode()[0],inplace=True)
df_test['BsmtExposure'].fillna(value=df_train['BsmtExposure'].mode()[0],inplace=True)

df_test['BsmtFinType1'].fillna(value=df_train['BsmtFinType1'].mode()[0],inplace=True)
df_test['BsmtFinType2'].fillna(value=df_train['BsmtFinType2'].mode()[0],inplace=True)
df_test['KitchenQual'].fillna(value=df_train['KitchenQual'].mode()[0],inplace=True)
df_test['GarageType'].fillna(value=df_train['GarageType'].mode()[0],inplace=True)
df_test['GarageCond'].fillna(value=df_train['GarageCond'].mode()[0],inplace=True)
df_test['GarageFinish'].fillna(value=df_train['GarageFinish'].mode()[0],inplace=True)
df_test['GarageQual'].fillna(value=df_train['GarageQual'].mode()[0],inplace=True)

#handling categorical data in test

temp_test=cat_test

df_dummy_test=pd.get_dummies(df_test[temp_test],prefix=temp_test,drop_first=True)

df_final_test=pd.concat([df_test[non_cat_test],df_dummy_test],axis=1)

X_test=df_final_test

missed_col=[]
for col in X_train.columns:
    if col not in X_test.columns:
        missed_col.append(col)

for i in missed_col:
    df_final_test[i]=0

X_test=df_final_test
y_predict=RFC.predict(X_test)
df_submit = pd.DataFrame({'Id':df_test['Id'],'SalePrice':y_predict})
df_submit.to_csv('Final Submission.csv',index=False)


