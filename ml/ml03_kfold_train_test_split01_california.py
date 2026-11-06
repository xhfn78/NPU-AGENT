import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test=train_test_split(
   x,y, shuffle=True,train_size=0.8,random_state=333
)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#1. data


print(x.shape,y.shape)

n_split = 5

kFold = KFold(n_splits=n_split,shuffle=True,random_state=333)


#2.모델

model = XGBRegressor()



#3,4.훈련,평가

scores = cross_val_score(
    model,
    x_train,
    y_train,
    cv=kFold,
    scoring='r2'
)

print('K-Fold R2:', scores)
print('평균 R2:', scores.mean())
y_predict = cross_val_predict(model,x_test,y_test, cv=kFold)

r2 = r2_score(y_test,y_predict)
print('r2:',r2)

# DecisionTreeRegressor
# r2 : [0.60096793 0.60376131 0.58624219 0.61533808 0.62028893] 
#  평균 r2:  0.61


# RandomForestRegressor
# r2 : [0.80301476 0.80816173 0.79648235 0.82036212 0.82122363] 
#  평균 r2:  0.81

# XGBRegressor
#r2 : [0.82742129 0.84140555 0.82123327 0.84284717 0.8478747 ] 
#  평균 r2:  0.84


# K-Fold R2: [0.83647596 0.84005549 0.82732296 0.83453094 0.83432637]
# 평균 R2: 0.8345423425160268
# r2: 0.7742655784877135



