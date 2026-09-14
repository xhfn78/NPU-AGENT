# [실습] Scaler 4종 비교 - 손글씨 숫자 (다중 분류)
#
# keras27과 달라진 점: 스케일러를 x_train에만 fit한다.
# MinMax / Standard / MaxAbs / Robust 중 하나만 주석을 풀어서 쓰고,
# 결과가 어떻게 달라지는지 아래 기록과 비교해본다.
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense 
import time
from sklearn.metrics import accuracy_score

#1데이터

datasets = load_digits()
# print(datasets)


x = datasets.data
y = datasets.target
# print(x.shape,y.shape) #(1797, 64) (1797,)

y = pd.get_dummies(y,dtype=int)
# print(y)

# print(np.unique(y))  #[0 1]
# print(np.unique(y,return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))
# print(pd.Series(y).value_counts())

x_train,x_test,y_train,y_test =train_test_split(
    x,y,
    random_state=333,
    train_size=0.8,
    stratify=y,  #x,y데이터를 나눌떄 stratify=y이걸안넣으면 x,y서로 데이터 크기가 달랐을떄 비율편차가 생길수있음
)
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler 
from sklearn.preprocessing import RobustScaler
##############################################################################
# scaler = MinMaxScaler()
##############################################################################
# MinMaxScaler
#   X_scaled = (X - X_min) / (X_max - X_min)
#   fit한 데이터의 최솟값을 0, 최댓값을 1로 만든다.
#   단점: 이상치가 하나라도 있으면 Min/Max 자체가 그 이상치로 잡혀서 크게 흔들린다.



##############################################################################
# scaler = StandardScaler()
##############################################################################
# StandardScaler
#   z = (x - 평균) / 표준편차
#   평균을 0, 표준편차를 1로 만든다.
#   z = 1 은 "평균보다 표준편차 1개만큼 위"라는 뜻이고, z = -2 는 "평균보다 2개만큼 아래"라는 뜻이다.
#   단점: 평균과 표준편차도 이상치의 영향을 받는다.



##############################################################################
# scaler = MaxAbsScaler()
##############################################################################
# MaxAbsScaler
#   X_scaled = X / max(|X|)
#   그 feature의 최대 절댓값으로 나눈다. 예) [-50, 0, 100] → [-0.5, 0, 1.0]
#   주의: 최솟값이 항상 -1이 되는 게 아니라, 절댓값이 가장 큰 값만 ±1이 된다.
#   단점: 최대 절댓값을 기준으로 삼기 때문에 큰 이상치에 민감하다.


##############################################################################
scaler = RobustScaler()
##############################################################################
# 이상치에 강력함
# RobustScaler
#   X_scaled = (X - 중앙값) / IQR       (IQR = 3사분위수 - 1사분위수)
#   중심을 평균 대신 중앙값으로, 폭을 표준편차 대신 IQR로 잡는다.
#   중앙값과 IQR은 이상치 하나에 잘 흔들리지 않아서 이상치에 강하다.
#   단, 이상치를 제거하는 게 아니라 이상치 때문에 스케일링 기준이 왜곡되는 걸 줄이는 것이다.
# scaler.fit(x_train)   # fit만 하는 줄. 아래에서 fit_transform으로 한 번에 하므로 중복이라 꺼둔다.
# [ 스케일러는 x_train에만 fit한다 ]
#   x_test와 실전 데이터는 x_train에서 학습한 기준으로 transform만 해야 한다.
#   test 데이터의 정보가 스케일러에 미리 반영되면 평가를 믿을 수 없게 되기 때문이다.
x_train = scaler.fit_transform(x_train)   # x_train 기준을 학습(fit)하고 동시에 변환(transform)
x_test = scaler.transform(x_test)         # test는 transform만 (fit 금지)

#2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=64, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(30, activation= 'relu'))
model.add(Dense(10,activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics =['acc']
              )
es = EarlyStopping(
    monitor= 'val_loss',
    mode= 'auto',
    patience=200,
    restore_best_weights=True,
)
start_time =time.time()
model.fit(x_train,y_train, epochs=500,batch_size=300,
          verbose=1,
          validation_split=0.3,
          callbacks =[es],
          )
end_time =time.time()

result = model.evaluate(x_test,y_test,)
print('loss: ',result[0])
print('acc: ',round(result[1],2))
y_predict= model.predict(x_test) 

y_predict = np.argmax(y_predict,axis=1) 
# print(y_predict)#[0 2 0 1 1 2 0 2 0 2 2 1 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 2]
y_test = np.argmax(y_test, axis=1)
# print(y_test) #[0 2 0 1 1 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 1]
# #######################################################
# y_predict = np.argmax(model.predict(x_test),axis =1)
# y_test_argmax =np.argmax(y_test,axis=1)
# ########################################################
# y_predict = model.predict(x_test)
# print(y_predict)


# 주의: 함수 이름(accuracy_score)을 변수 이름으로 덮어쓰면 안 되므로 acc_score로 받는다.
acc_score = accuracy_score(y_test,y_predict)  
#지금까지는 y_predict 값은 [0.7,0.2,0.1]이런식으로 되어있어서 비교가 불가능함 >>가장큰 수를 1로 바꿔줘야함 그래서 결과를 [1,0,0]으로 변경후 비교 
print('acc_score :',acc_score)
print('걸린시간: ', round(end_time-start_time, 2),'초')

"""
2차시도----스케일러 적용----
random : 333
train_size = 0.8
epochs = 500
batch_size = 300
결과
loss:  0.10323592275381088
acc:  0.96
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
acc_score : 0.9555555555555556
걸린시간:  8.13 초
"""
'''
# 3차시도
# loss:  0.13404762744903564
# acc:  0.97
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
# acc_score : 0.9694444444444444
# 걸린시간:  12.46 초

#4차시도
# loss:  0.09416898339986801
# acc:  0.97
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
# acc_score : 0.9666666666666667
# 걸린시간:  17.67 초
'''


'''
# 5차시도--standard-
loss:  0.09308037161827087
acc:  0.97
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
acc_score : 0.9666666666666667
걸린시간:  15.6 초

'''