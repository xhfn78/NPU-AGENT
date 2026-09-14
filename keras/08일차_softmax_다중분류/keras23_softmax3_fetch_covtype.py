# [실습] 다중 분류 - 산림 수종(fetch_covtype) 데이터셋
# 클래스가 7개이고 행이 581,012개인 큰 데이터다.
# 여기서 to_categorical의 함정을 직접 만나게 된다.
from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense 
import time
from sklearn.metrics import accuracy_score


#1.데이터
datasets = fetch_covtype()
# print(datasets)
# shape=(581012, 54))  ,shape=(581012,)

x = datasets.data
y = datasets.target

# print(x.shape,y.shape)  #(581012, 54) (581012,)
print(np.unique(y,return_counts=True)) 

# from tensorflow.keras.utils import to_categorical 
# y = to_categorical(y)
# print(y)
# print(y.shape) #(581012, 8)
'''
#(array([1, 2, 3, 4, 5, 6, 7], dtype=int32),
#array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))
>>>>.
to_categorical  쓰면 0~부터 컬럼을 만들어서 만약 1,2,3,4,5,6,7의 컬럼이 형성되어있으면 
0,1,2,3,4,5,6,7,8로 늘어남 

즉 이 데이터의 클래스는 1~7 (0은 없음)인데,
to_categorical은 0번 클래스가 있다고 가정해서 8열짜리를 만들어버린다. (581012, 8)
아무도 쓰지 않는 0번 열이 하나 끼어드는 셈이다.

반면 pandas의 get_dummies나 sklearn의 OneHotEncoder는
실제로 존재하는 1~7에 대해서만 7열을 만든다. (581012, 7)
그래서 여기서는 get_dummies를 쓴다.
'''
# dtype=int를 안 붙이면 1/0 대신 True/False로 나온다.
# 더 안전하게 하려면 .values를 붙여 넘파이 배열로 꺼내 쓴다.
y = pd.get_dummies(y,dtype=int)
# print(y.shape) #(581012, 7)


x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    train_size=0.7,
    random_state=333,
    shuffle=True,
    stratify=y,
    )

# exit()   # 여기서 멈추고 데이터 모양만 확인하려고 썼던 줄. 켜두면 아래가 실행되지 않는다.

#2. 모델구성
model = Sequential()
model.add(Dense(200, input_dim=54, activation= 'relu'))
model.add(Dense(300, activation= 'relu'))
model.add(Dense(400,activation= 'relu'))
model.add(Dense(300, activation= 'relu'))
model.add(Dense(200, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(7,activation='softmax'))  # 클래스가 7개이므로 출력도 7개

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics =['acc']
              )
es = EarlyStopping(
    monitor= 'val_loss',
    mode= 'auto',
    patience=100,
    restore_best_weights=True,
)
start_time =time.time()
model.fit(x_train,y_train, epochs=2000,batch_size=25000,
          verbose=1,
          validation_split=0.3,
          callbacks =[es],
          )
end_time =time.time()

#4. 평가, 예측
result = model.evaluate(x_test,y_test,)
print('loss: ',result[0])
print('acc: ',round(result[1],2))
y_predict= model.predict(x_test) 

y_predict = np.argmax(y_predict,axis=1) 
print(y_predict)#[0 2 0 1 1 2 0 2 0 2 2 1 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 2]
# get_dummies로 만든 열 인덱스는 0~6이고, 예측값도 0~6으로 나온다.
# 실제 클래스는 1~7이지만 인덱스끼리 비교하는 것이라 정확도 계산에는 문제가 없다.
y_test = np.argmax(y_test, axis=1)
print(y_test) #[0 2 0 1 1 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 1]
# #######################################################
# y_predict = np.argmax(model.predict(x_test),axis =1)
# y_test_argmax =np.argmax(y_test,axis=1)
# ########################################################
# y_predict = model.predict(x_test)
# print(y_predict)


# 주의: accuracy_score = accuracy_score(...) 처럼 쓰면
# 함수 이름이 숫자로 덮어써져서 다음에 그 함수를 못 쓰게 된다. 변수 이름은 다르게 둔다.
acc_score = accuracy_score(y_test,y_predict)  
#지금까지는 y_predict 값은 [0.7,0.2,0.1]이런식으로 되어있어서 비교가 불가능함 >>가장큰 수를 1로 바꿔줘야함 그래서 결과를 [1,0,0]으로 변경후 비교 
print('acc_score :',acc_score)
print('걸린시간: ', round(end_time-start_time, 2),'초')