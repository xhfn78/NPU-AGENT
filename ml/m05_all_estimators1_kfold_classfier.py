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

n_split = 5

kFold = KFold(n_splits=n_split,shuffle=True,random_state=333)


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

        scores = cross_val_score(
        model,
        x_train,
        y_train,
        cv=kFold,
        scoring='accuracy'
        )

        print('K-Fold acc:', scores)
        print('평균 acc:', scores.mean())
        y_predict = cross_val_predict(model,x_test,y_test, cv=kFold)
        acc = accuracy_score(y_test,y_predict)
        print('acc:',acc)

    except:
        print(name,'모델 오류')


# sk version 1.9.1
# 모델의 갯수: 44
# K-Fold acc: [0.91666667 0.91666667 0.95833333 1.         0.91666667]
# 평균 acc: 0.9416666666666667
# acc: 0.9
# K-Fold acc: [0.95833333 0.95833333 0.95833333 1.         0.91666667]
# 평균 acc: 0.9583333333333334
# acc: 0.9333333333333333
# K-Fold acc: [0.66666667 0.75       0.70833333 0.75       0.95833333]
# 평균 acc: 0.7666666666666667
# acc: 0.6333333333333333
# K-Fold acc: [0.83333333 0.875      0.875      0.95833333 0.875     ]
# 평균 acc: 0.8833333333333334
# acc: 0.6666666666666666
# CategoricalNB 모델 오류
# ClassifierChain 모델 오류
# ComplementNB 모델 오류
# K-Fold acc: [1.         0.95833333 0.95833333 1.         0.95833333]
# 평균 acc: 0.975
# acc: 0.9333333333333333
# K-Fold acc: [0.20833333 0.20833333 0.29166667 0.20833333 0.20833333]
# 평균 acc: 0.225
# acc: 0.16666666666666666
# K-Fold acc: [0.70833333 1.         0.91666667 1.         0.95833333]
# 평균 acc: 0.9166666666666666
# acc: 0.9
# K-Fold acc: [0.95833333 0.95833333 0.95833333 1.         0.95833333]
# 평균 acc: 0.9666666666666666
# acc: 0.9333333333333333
# FixedThresholdClassifier 모델 오류
# K-Fold acc: [0.875      0.95833333 0.95833333 1.         0.95833333]
# 평균 acc: 0.95
# acc: 0.8666666666666667
# K-Fold acc: [0.91666667 0.95833333 0.91666667 1.         0.95833333]
# 평균 acc: 0.95
# acc: 0.8666666666666667
# K-Fold acc: [0.95833333 0.95833333 0.95833333 1.         0.95833333]
# 평균 acc: 0.9666666666666666
# acc: 0.9
# K-Fold acc: [0.91666667 0.91666667 0.95833333 1.         0.91666667]
# 평균 acc: 0.9416666666666667
# acc: 0.16666666666666666
# K-Fold acc: [1.         0.95833333 0.83333333 1.         0.95833333]
# 평균 acc: 0.95
# acc: 0.8666666666666667
# K-Fold acc: [0.91666667 0.95833333 0.875      1.         0.95833333]
# 평균 acc: 0.9416666666666667
# acc: 0.9
# K-Fold acc: [0.91666667 0.95833333 0.875      1.         0.95833333]
# 평균 acc: 0.9416666666666667
# acc: 0.9
# K-Fold acc: [0.95833333 1.         0.95833333 1.         0.95833333]
# 평균 acc: 0.975
# acc: 0.9
# K-Fold acc: [0.91666667 1.         0.95833333 1.         0.91666667]
# 평균 acc: 0.9583333333333334
# acc: 0.8
# K-Fold acc: [0.95833333 0.95833333 0.91666667 1.         0.95833333]
# 평균 acc: 0.9583333333333334
# acc: 0.8333333333333334
# K-Fold acc: [0.95833333 1.         0.91666667 1.         0.95833333]
# 평균 acc: 0.9666666666666666
# acc: 0.9
# K-Fold acc: [0.95833333 0.95833333 0.875      1.         0.95833333]
# 평균 acc: 0.95
# acc: 0.8333333333333334
# MultiOutputClassifier 모델 오류
# MultinomialNB 모델 오류
# K-Fold acc: [0.75       0.83333333 0.91666667 1.         0.83333333]
# 평균 acc: 0.8666666666666666
# acc: 0.7666666666666667
# K-Fold acc: [0.95833333 0.95833333 0.91666667 1.         0.95833333]
# 평균 acc: 0.9583333333333334
# acc: 0.8666666666666667
# OneVsOneClassifier 모델 오류
# OneVsRestClassifier 모델 오류
# OutputCodeClassifier 모델 오류
# K-Fold acc: [0.91666667 0.875      0.91666667 1.         0.91666667]
# 평균 acc: 0.925
# acc: 0.7
# K-Fold acc: [0.875      0.79166667 0.83333333 0.875      0.75      ]
# 평균 acc: 0.825
# acc: 0.7666666666666667
# K-Fold acc: [0.95833333 1.         0.83333333 1.         1.        ]
# 평균 acc: 0.9583333333333334
# acc: 0.8666666666666667
# K-Fold acc: [0.91666667 0.91666667        nan 1.         0.91666667]
# 평균 acc: nan
# RadiusNeighborsClassifier 모델 오류
# K-Fold acc: [0.95833333 0.95833333 0.95833333 1.         0.95833333]
# 평균 acc: 0.9666666666666666
# acc: 0.9
# K-Fold acc: [0.79166667 0.83333333 0.83333333 0.91666667 0.79166667]
# 평균 acc: 0.8333333333333334
# acc: 0.7
# K-Fold acc: [0.79166667 0.83333333 0.83333333 0.91666667 0.79166667]
# 평균 acc: 0.8333333333333334
# acc: 0.7
# K-Fold acc: [0.91666667 0.95833333 0.875      0.875      0.95833333]
# 평균 acc: 0.9166666666666666
# acc: 0.6666666666666666
# K-Fold acc: [0.95833333 1.         0.875      1.         0.95833333]
# 평균 acc: 0.9583333333333334
# acc: 0.9
# SelfTrainingClassifier 모델 오류
# StackingClassifier 모델 오류
# TunedThresholdClassifierCV 모델 오류
# VotingClassifier 모델 오류        


