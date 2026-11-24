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
all = all_estimators(type_filter='classifier',)
# all = all_estimators(type_filter='regressor',)
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
        acc= model.score(x_test,y_test)
        print(name,'의 정답률:' ,round(acc,2))
    except:
        print(name,'모델 오류')



# sk version 1.9.1
# 모델의 갯수: 44
# AdaBoostClassifier 의 정답률: 0.9
# BaggingClassifier 의 정답률: 0.9
# BernoulliNB 의 정답률: 0.7
# CalibratedClassifierCV 의 정답률: 0.83
# CategoricalNB 모델 오류
# ClassifierChain 모델 오류
# ComplementNB 모델 오류
# DecisionTreeClassifier 의 정답률: 0.9
# DummyClassifier 의 정답률: 0.33
# ExtraTreeClassifier 의 정답률: 0.9
# ExtraTreesClassifier 의 정답률: 0.9
# FixedThresholdClassifier 모델 오류
# GaussianNB 의 정답률: 0.93
# GaussianProcessClassifier 의 정답률: 0.9
# GradientBoostingClassifier 의 정답률: 0.93
# HistGradientBoostingClassifier 의 정답률: 0.9
# KNeighborsClassifier 의 정답률: 0.9
# LabelPropagation 의 정답률: 0.9
# LabelSpreading 의 정답률: 0.9
# LinearDiscriminantAnalysis 의 정답률: 0.97
# LinearSVC 의 정답률: 0.9
# LogisticRegression 의 정답률: 0.93
# LogisticRegressionCV 의 정답률: 0.93
# MLPClassifier 의 정답률: 0.93
# MultiOutputClassifier 모델 오류
# MultinomialNB 모델 오류
# NearestCentroid 의 정답률: 0.83
# NuSVC 의 정답률: 0.93
# OneVsOneClassifier 모델 오류
# OneVsRestClassifier 모델 오류
# OutputCodeClassifier 모델 오류
# PassiveAggressiveClassifier 의 정답률: 0.93
# Perceptron 의 정답률: 0.83
# QuadraticDiscriminantAnalysis 의 정답률: 0.97
# RadiusNeighborsClassifier 의 정답률: 0.93
# RandomForestClassifier 의 정답률: 0.93
# RidgeClassifier 의 정답률: 0.8
# RidgeClassifierCV 의 정답률: 0.8
# SGDClassifier 의 정답률: 0.77
# SVC 의 정답률: 0.93
# SelfTrainingClassifier 모델 오류
# StackingClassifier 모델 오류
# TunedThresholdClassifierCV 모델 오류
# VotingClassifier 모델 오류