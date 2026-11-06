import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
import pandas as pd

#1. 데이터
path = './_data/kaggle_bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv)
submission = pd.read_csv(path +'sampleSubmission.csv',index_col=0)
# print(submission)



################# x,y 분리 ##########################
x = train_csv.drop(['casual','registered', 'count'], axis=1)
# print(x) #[10886 rows x 8 columns]

y = train_csv['count']
# print(y, y.shape)  ##(10886,)

#1. data


print(x.shape,y.shape)

n_split = 5

kFold = KFold(n_splits=n_split,shuffle=True,random_state=333)


#2.모델

model = XGBRegressor()



#3,4.훈련,평가

scores = cross_val_score(model, x,y, cv=kFold)

print('r2 :' ,scores ,'\n 평균 r2: ', round(np.mean(scores),2))

# (10886, 8) (10886,)
# r2 : [0.33359152 0.31047136 0.33931798 0.33344597 0.33251733] 
#  평균 r2:  0.33


