# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:20:03 2017

@author: huangshizhi
机器学习系列(12)_XGBoost参数调优完全指南（附Python代码）
http://blog.csdn.net/han_xiaoyang/article/details/52665396
"""

import pandas as pd
import time
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

train = pd.read_csv(r'D:\git\xgboost\tuning_params\Dataset\train_modified.csv',encoding='utf-8')
test =pd.read_csv(r'D:\git\xgboost\tuning_params\Dataset\test_modified.csv',encoding='utf-8')
target = 'Disbursed'
IDcol = 'ID'
#train['Disbursed'].value_counts()

'''
建模与交叉验证
写一个大的函数完成以下的功能
1. 数据建模
2. 求训练准确率
3. 求训练集AUC
4. 根据xgboost交叉验证更新n_estimators
5. 画出特征的重要度

'''
def modelfit(alg, dtrain, dtest,predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    alg = xgb4
    dtrain = train    
    start_time = time.time()    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        #xgtest = xgb.DMatrix(dtest[predictors].values)        
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #建模
    alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')
    
    #对训练集预测
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    end_time = time.time()
    print("运行时间为%.2f s" %(end_time-start_time))
    return dtrain_predictions
    
    
#Choose all predictors except target & IDcols
#第一步：确定学习速率和tree_based 参数调优的估计器数目    
predictors = [x for x in train.columns if x not in [target,IDcol]]
xgb1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
#模型训练
df_pred = modelfit(xgb1, train, predictors)


#第二步： max_depth 和 min_weight 参数调优
#参数范围

#train['Disbursed'].value_counts()

param_test1 = {
 'max_depth':list(range(3,10,2)),
 'min_child_weight':list(range(1,6,2))
}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
                                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
                       param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])

'''
GridSearchCV(cv=5, error_score='raise',
       estimator=XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=5,
       min_child_weight=1, missing=None, n_estimators=140, nthread=4,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=27, silent=True, subsample=0.8),
       fit_params={}, iid=False, n_jobs=4,
       param_grid={'max_depth': [3, 5, 7, 9], 'min_child_weight': [1, 3, 5]},
       pre_dispatch='2*n_jobs', refit=True, scoring='roc_auc', verbose=0)

{'max_depth': 5, 'min_child_weight': 1}

'''
gsearch1.grid_scores_ 
gsearch1.best_params_
gsearch1.best_score_

#对于max_depth和min_child_weight查找最好的参数
param_test2 = {
    'max_depth':[4,5,6],
    'min_child_weight':[4,5,6]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
                                        min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                       param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch2.fit(train[predictors],train[target])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

#交叉验证对min_child_weight寻找最合适的参数
param_test2b = {
    'min_child_weight':[4,6,8,10,12]
}
gsearch2b = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=4,
                                        min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                       param_grid = param_test2b, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2b.fit(train[predictors],train[target])

gsearch2b.grid_scores_, gsearch2b.best_params_, gsearch2b.best_score_

#max_depth=5,min_child_weight=4参数最优，与原代码有差异

param_test3 = {
    'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
                                        min_child_weight=4, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                       param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])

gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


predictors = [x for x in train.columns if x not in [target, IDcol]]
xgb2 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
modelfit(xgb2, train, predictors)

#0.883777(max_depth=4,min_child_weight=6, gamma=0)
#(max_depth=5,min_child_weight=1, gamma=0.1)

#对subsample 和 colsample_bytree用grid search寻找最合适的参数
param_test4 = {
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=5,
                                        min_child_weight=1, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                       param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
#{'colsample_bytree': 0.9, 'subsample': 0.9}
# 同上

param_test5 = {
    'subsample':[i/100.0 for i in range(85,95,5)],
    'colsample_bytree':[i/100.0 for i in range(80,100,5)]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=5,
                                        min_child_weight=1, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                       param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch5.fit(train[predictors],train[target])

gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

#对reg_alpha用grid search寻找最合适的参数
param_test6 = {
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=5,
                                        min_child_weight=1, gamma=0.1, subsample=0.9, colsample_bytree=0.9,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                       param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch6.fit(train[predictors],train[target])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_

xgb3 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.00001,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

modelfit(xgb3, train, predictors)
#AUC Score 0.894727
xgb4 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=5000,
        max_depth=5,
        min_child_weight=1,
        gamma=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.00001,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

h = modelfit(xgb4, train, predictors)
#AUC Score (Train): 0.887619




