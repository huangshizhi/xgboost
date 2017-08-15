# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 11:43:23 2017

@author: huangshizhi

https://github.com/aarshayj/Analytics_Vidhya/blob/master/Articles/Parameter_Tuning_GBM_with_Example/data_preparation.ipynb
"""

import pandas as pd
import numpy as np

#load data;加载数据，原始数据有一行乱码的
train = pd.read_csv(r'D:\git\xgboost\tuning_params\Dataset\Train_nyOWmfK.csv',encoding='utf-8')
test = pd.read_csv(r'D:\git\xgboost\tuning_params\Dataset\Test_bCtAN1w.csv',encoding='utf-8')

#显示dataframe列的类型
train.dtypes

#Combine into data:
train['source']= 'train'
test['source'] = 'test'
data=pd.concat([train, test],ignore_index=True)

#查看缺失值,Check missing
null_count = data.apply(lambda x: sum(x.isnull()))

'''
查看变量类型,Look at categories of all object variables

'''
var = ['Gender','Salary_Account','Mobile_Verified','Var1','Filled_Form','Device_Type','Var2','Source']
for v in var:
    print ('\nFrequency count for variable %s'%v)
    print (data[v].value_counts())

    
#Handle Individual Variables
len(data['City'].unique())
#drop city because too many unique
data.drop('City',axis=1,inplace=True)

data['DOB'].head()
    

#Create age variable,年龄 = 2015 - 出生年份
data['Age'] = data['DOB'].apply(lambda x: 115 - int(x[-2:]))
data['Age'].head()    
#drop DOB:
data.drop('DOB',axis=1,inplace=True)    
    
#data.boxplot(column=['Device_Type'],return_type='axes')

#Majority values missing so I'll create a new variable stating whether this is missing or note:
data['EMI_Loan_Submitted_Missing'] = data['EMI_Loan_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)
data[['EMI_Loan_Submitted','EMI_Loan_Submitted_Missing']].head(10)
#drop original vaiables:
data.drop('EMI_Loan_Submitted',axis=1,inplace=True)

len(data['Employer_Name'].value_counts())
#I'll drop the variable because too many unique values. Another option could be to categorize them manually
data.drop('Employer_Name',axis=1,inplace=True)

data['Existing_EMI'].describe()

#Impute by median (0) because just 111 missing:
data['Existing_EMI'].fillna(0, inplace=True)

#Majority values missing so I'll create a new variable stating whether this is missing or note:
data['Interest_Rate_Missing'] = data['Interest_Rate'].apply(lambda x: 1 if pd.isnull(x) else 0)
data[['Interest_Rate','Interest_Rate_Missing']].head(10)
#data['Interest_Rate'].value_counts()
#data['Interest_Rate'].describe()
data.drop('Interest_Rate',axis=1,inplace=True)




#Drop this variable because doesn't appear to affect much intuitively
data.drop('Lead_Creation_Date',axis=1,inplace=True)
#Impute with median because only 111 missing:
data['Loan_Amount_Applied'].fillna(data['Loan_Amount_Applied'].median(),inplace=True)
data['Loan_Tenure_Applied'].fillna(data['Loan_Tenure_Applied'].median(),inplace=True)

#High proportion missing so create a new var whether present or not
data['Loan_Amount_Submitted_Missing'] = data['Loan_Amount_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)
data['Loan_Tenure_Submitted_Missing'] = data['Loan_Tenure_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)
#Remove old vars
data.drop(['Loan_Amount_Submitted','Loan_Tenure_Submitted'],axis=1,inplace=True)
data.drop('LoggedIn',axis=1,inplace=True)
#Salary account has mnay banks which have to be manually grouped
data.drop('Salary_Account',axis=1,inplace=True)
#High proportion missing so create a new var whether present or not
data['Processing_Fee_Missing'] = data['Processing_Fee'].apply(lambda x: 1 if pd.isnull(x) else 0)
#drop old
data.drop('Processing_Fee',axis=1,inplace=True)
data['Source'] = data['Source'].apply(lambda x: 'others' if x not in ['S122','S133'] else x)
data['Source'].value_counts()


'''
最终的数据
'''
#d1 = data[data['Var1'].isnull()]

data.apply(lambda x: sum(x.isnull()))
data.dtypes

#Numerical Coding:数值编码
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var_to_encode = ['Device_Type','Filled_Form','Gender','Var1','Var2','Mobile_Verified','Source']
for col in var_to_encode:
    data[col] = le.fit_transform(data[col])


data = pd.get_dummies(data, columns=var_to_encode)

data.columns

#拆分成训练集和测试集
train = data.loc[data['source']=='train']
test = data.loc[data['source']=='test']


train.drop('source',axis=1,inplace=True)
test.drop(['source','Disbursed'],axis=1,inplace=True)

train.to_csv(r'D:\git\xgboost\tuning_params\Dataset\train_modified.csv',index=False)
test.to_csv(r'D:\git\xgboost\tuning_params\Dataset\test_modified.csv',index=False)






    
