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



##############################################################################
# scaler = StandardScaler()
##############################################################################



##############################################################################
# scaler = MaxAbsScaler()
##############################################################################


##############################################################################
scaler = RobustScaler()
##############################################################################
scaler.fit(x_train) # x 값을  MinMaxScaler으로 실행시킬 준비
x_train = scaler.fit_transform(x_train) # 0~1 값 변환 사이로변환
x_test = scaler.transform(x_test) 

#2.모델구성
model = Sequential()
model.add(Dense(30, input_dim=64, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(30, activation= 'relu'))
model.add(Dense(10,activation='softmax'))

#3.컴파일,훈련
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


accuracy_score =accuracy_score(y_test,y_predict)  
#지금까지는 y_predict 값은 [0.7,0.2,0.1]이런식으로 되어있어서 비교가 불가능함 >>가장큰 수를 1로 바꿔줘야함 그래서 결과를 [1,0,0]으로 변경후 비교 
print('acc_score :',accuracy_score)
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