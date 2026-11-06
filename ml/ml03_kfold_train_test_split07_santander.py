import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


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
x_train,x_test,y_train,y_test=train_test_split(
   x,y, shuffle=True,train_size=0.8,random_state=333,stratify=y,
)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



n_split = 5

kFold = KFold(n_splits=n_split,shuffle=True,random_state=333)


#2.모델

model = XGBClassifier()



#3,4.훈련,평가


from sklearn.model_selection import cross_val_score
scores = cross_val_score(
    model,
    x_train,
    y_train,
    cv=kFold,
    scoring='accuracy'
)

print('K-Fold Accuracy:', scores)
print('평균 Accuracy:', scores.mean())
y_predict = cross_val_predict(model,x_test,y_test, cv=kFold)

acc_score = accuracy_score(y_test,y_predict)
print('acc:',acc_score)



# K-Fold Accuracy: [0.9120625  0.91334375 0.9120625  0.91346875 0.9133125 ]
# 평균 Accuracy: 0.9128499999999999
# acc: 0.908025


