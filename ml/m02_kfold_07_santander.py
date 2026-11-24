import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
import pandas as pd

#1. 데이터
path = './_data/kaggle_santander/'   #<<< 상대경로 (윈도우/맥 어디서나 동작)
# path = 'c://study//_data//kaggle_santander//'   #<<< 윈도우 절대경로. // 두 개 써도 가능하지만 맥에서는 안 됨
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col=0 )
# print(test_csv)
submission_csv = pd.read_csv(path +'sample_submission.csv',index_col=0)

x = train_csv.drop(['target'], axis=1)  #타겟 열 한개뺴기
# # print(x) #[10886 rows x 8 columns]

y = train_csv['target']
#1. data



n_split = 5

kFold = KFold(n_splits=n_split,shuffle=True,random_state=333)


#2.모델

model = DecisionTreeClassifier()



#3,4.훈련,평가

scores = cross_val_score(model, x,y, cv=kFold)

print('acc :' ,scores ,'\n 평균 acc: ', round(np.mean(scores),2))

# acc : [0.837725 0.835075 0.834525 0.83245  0.8368  ] 
#  평균 acc:  0.84



