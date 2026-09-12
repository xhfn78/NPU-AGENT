from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense 
import time
from sklearn.metrics import accuracy_score


#1데이터 
datasets = load_wine()
print(datasets)
x = datasets.data
y = datasets.target
print(x.shape,y.shape)  #(178, 13) (178,)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# print(y)
# print(y.shape) #(178, 3)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    train_size=0.8,
    random_state=333,
    shuffle=True,
    stratify=y,
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
model.add(Dense(10, input_dim=13, activation= 'relu'))
model.add(Dense(20, activation= 'relu'))
model.add(Dense(30,activation= 'relu'))
model.add(Dense(40, activation= 'relu'))
model.add(Dense(30, activation= 'relu'))
model.add(Dense(3,activation='softmax'))

#3.컴파일,훈련
model.compile(loss= 'categorical_crossentropy' , 
              optimizer='adam', 
              metrics= ['acc']
              )
es = EarlyStopping(
    monitor= 'val_loss',
    mode= 'auto',
    patience=30,
    restore_best_weights=True,
)
start_time =time.time()
model.fit(x_train,y_train, epochs=1000,batch_size=8,
          verbose=1,
          validation_split=0.2,
          callbacks =[es],
          )
end_time =time.time()
#4.평가,예측
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

'''
# loss:  0.13087505102157593
# acc:  0.94
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step
# acc_score : 0.9444444444444444
# 걸린시간:  14.58 초
'''


#MinMaxScaler
'''
loss:  0.16965200006961823
acc:  0.94
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step
acc_score : 0.9444444444444444
걸린시간:  8.62 초
'''


#StandardScaler
'''
loss:  0.11229284852743149
acc:  0.97
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step
acc_score : 0.9722222222222222
걸린시간:  4.44 초
'''

##maxabs
'''
loss:  0.18491190671920776
acc:  0.97
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 41ms/step
acc_score : 0.9722222222222222
걸린시간:  19.49 초
'''


