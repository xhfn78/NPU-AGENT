# [실습] 다중 분류 - 와인(wine) 데이터셋
# iris와 똑같은 3개 클래스 분류. 원-핫은 to_categorical로 해본다.
from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense 
import time
from sklearn.metrics import accuracy_score


#1. 데이터
datasets = load_wine()
print(datasets)
x = datasets.data
y = datasets.target
print(x.shape,y.shape)  #(178, 13) (178,)

# 원-핫 방법 1. 텐서플로 to_categorical
# wine의 클래스는 0, 1, 2 라서 0번부터 시작한다.
# to_categorical은 0번부터 열을 만들기 때문에 이 경우엔 딱 3열이 되어 문제가 없다.
# (클래스가 1부터 시작하는 데이터에서는 0번 자리가 하나 더 생기는 함정이 있다 → keras23_softmax3 참고)
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
# acc= 0.95

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=13, activation= 'relu'))
model.add(Dense(20, activation= 'relu'))
model.add(Dense(30,activation= 'relu'))
model.add(Dense(40, activation= 'relu'))
model.add(Dense(30, activation= 'relu'))
model.add(Dense(3,activation='softmax'))  # 클래스가 3개이므로 출력도 3개, 다중분류는 softmax

#3. 컴파일, 훈련
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
#4. 평가, 예측
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


# 주의: accuracy_score = accuracy_score(...) 처럼 쓰면
# 함수 이름이 숫자로 덮어써져서 다음에 그 함수를 못 쓰게 된다. 변수 이름은 다르게 둔다.
acc_score = accuracy_score(y_test,y_predict)  
#지금까지는 y_predict 값은 [0.7,0.2,0.1]이런식으로 되어있어서 비교가 불가능함 >>가장큰 수를 1로 바꿔줘야함 그래서 결과를 [1,0,0]으로 변경후 비교 
print('acc_score :',acc_score)
print('걸린시간: ', round(end_time-start_time, 2),'초')

'''
# loss:  0.13087505102157593
# acc:  0.94
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step
# acc_score : 0.9444444444444444
# 걸린시간:  14.58 초
'''