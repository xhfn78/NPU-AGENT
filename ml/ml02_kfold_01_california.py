import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target



#1. data


print(x.shape,y.shape)

n_split = 5

kFold = KFold(n_splits=n_split,shuffle=True,random_state=333)


#2.모델

model = XGBRegressor()



#3,4.훈련,평가

scores = cross_val_score(model, x,y, cv=kFold)

print('r2 :' ,scores ,'\n 평균 r2: ', round(np.mean(scores),2))

# DecisionTreeRegressor
# r2 : [0.60096793 0.60376131 0.58624219 0.61533808 0.62028893] 
#  평균 r2:  0.61


# RandomForestRegressor
# r2 : [0.80301476 0.80816173 0.79648235 0.82036212 0.82122363] 
#  평균 r2:  0.81

# XGBRegressor
#r2 : [0.82742129 0.84140555 0.82123327 0.84284717 0.8478747 ] 
#  평균 r2:  0.84



