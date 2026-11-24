import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
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

x_train,x_test,y_train,y_test=train_test_split(
   x,y, shuffle=True,train_size=0.8,random_state=333
)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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


# K-Fold R2: [0.28897172 0.31962562 0.32887954 0.32639587 0.29020262]
# 평균 R2: 0.3108150720596313
# r2: 0.08696401119232178


