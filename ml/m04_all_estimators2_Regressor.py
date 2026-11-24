import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators
import sklearn as sk
import warnings

warnings.filterwarnings('ignore')  #warning 있는 거 무시하고 진행

#1. data

x,y = load_iris(return_X_y=True)
# print(x.shape,y.shape)

x_train,x_test, y_train,y_test = train_test_split(
    x,y ,shuffle=True,random_state=333,train_size=0.8,stratify=y,
)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2.모델
# all = all_estimators(type_filter='classifier',)
all = all_estimators(type_filter='regressor',)
print('all models:', all)
print('sk version',sk.__version__) # 1.9.1
print('모델의 갯수:',len(all)) #44  분류모델  , 회귀모델 55개

for name,model in all:
    # print(name)
    try:    
        model = model()

        #.3훈련
        model.fit(x_train,y_train)
        #4.평가
        r2= model.score(x_test,y_test)
        print(name,'의 정답률:' ,round(r2,2))
    except:
        print(name,'모델 오류')

# sk version 1.9.1
# 모델의 갯수: 55
# ARDRegression 의 정답률: 0.92
# AdaBoostRegressor 의 정답률: 0.9
# BaggingRegressor 의 정답률: 0.89
# BayesianRidge 의 정답률: 0.92
# CCA 모델 오류
# DecisionTreeRegressor 의 정답률: 0.85
# DummyRegressor 의 정답률: 0.0
# ElasticNet 의 정답률: 0.46
# ElasticNetCV 의 정답률: 0.92
# ExtraTreeRegressor 의 정답률: 0.85
# ExtraTreesRegressor 의 정답률: 0.91
# GammaRegressor 모델 오류
# GaussianProcessRegressor 의 정답률: -1.3
# GradientBoostingRegressor 의 정답률: 0.9
# HistGradientBoostingRegressor 의 정답률: 0.89
# HuberRegressor 의 정답률: 0.92
# IsotonicRegression 모델 오류
# KNeighborsRegressor 의 정답률: 0.94
# KernelRidge 의 정답률: -0.44
# Lars 의 정답률: 0.92
# LarsCV 의 정답률: 0.92
# Lasso 의 정답률: 0.0
# LassoCV 의 정답률: 0.92
# LassoLars 의 정답률: 0.0
# LassoLarsCV 의 정답률: 0.92
# LassoLarsIC 의 정답률: 0.92
# LinearRegression 의 정답률: 0.92
# LinearSVR 의 정답률: 0.92
# MLPRegressor 의 정답률: 0.95
# MultiOutputRegressor 모델 오류
# MultiTaskElasticNet 모델 오류
# MultiTaskElasticNetCV 모델 오류
# MultiTaskLasso 모델 오류
# MultiTaskLassoCV 모델 오류
# NuSVR 의 정답률: 0.93
# OrthogonalMatchingPursuit 의 정답률: 0.89
# OrthogonalMatchingPursuitCV 의 정답률: 0.92
# PLSCanonical 모델 오류
# PLSRegression 의 정답률: 0.92
# PassiveAggressiveRegressor 의 정답률: 0.86
# PoissonRegressor 의 정답률: 0.7
# QuantileRegressor 의 정답률: 0.0
# RANSACRegressor 의 정답률: 0.92
# RadiusNeighborsRegressor 의 정답률: 0.93
# RandomForestRegressor 의 정답률: 0.91
# RegressorChain 모델 오류
# Ridge 의 정답률: 0.92
# RidgeCV 의 정답률: 0.92
# SGDRegressor 의 정답률: 0.92
# SVR 의 정답률: 0.94
# StackingRegressor 모델 오류
# TheilSenRegressor 의 정답률: 0.92
# TransformedTargetRegressor 의 정답률: 0.92
# TweedieRegressor 의 정답률: 0.86
# VotingRegressor 모델 오류



