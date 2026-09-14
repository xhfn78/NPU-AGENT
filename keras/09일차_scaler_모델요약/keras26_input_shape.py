# 23-1 카피
# [실습] input_dim 대신 input_shape 쓰기
#
# input_dim은 1차원 입력(열 개수)만 표현할 수 있다.
# 이미지처럼 입력이 2차원, 3차원이 되면 input_dim으로는 못 적는다.
# 그래서 튜플로 적는 input_shape를 쓴다.
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense 
import time
from sklearn.metrics import accuracy_score


#1. 데이터
datasets = load_iris()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
# print(x.shape,y.shape) #(150, 4) (150,)

# print(y)

##################원 핫1, to_categorycal####################
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# # print(y)
# # print(y.shape)
##################원 핫2, pandas ####################

# y = pd.get_dummies(y,dtype=int)
# print(y)


# # ##################원 핫3, sklearn ####################

from sklearn.preprocessing import OneHotEncoder
#y = y.reshape(150,1)
y = y.reshape(-1,1)  #(150,1)
# print(y)
# print(y.shape) #(150, 1)
# exit()
# ohe = OneHotEncoder()  #sparse형태로 나온다(혼동행렬)
ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(y)
# print(y)
# print(y) #<Compressed Sparse Row sparse matrix of dtype 'float64'
#          with 150 stored elements and shape (150, 3)>

#reshape 조건 1.내용,2순서 (150,)>(150,1)  [1,2,3](3,)> [[1],[2],[3]] (3,1)




# exit()
# print(np.unique(y,return_counts=True)) #(array([0, 1, 2]), array([50, 50, 50])) #자주 많이씀
x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    train_size=0.8,
    random_state=333,
    shuffle=True,
    stratify=y,
    )
# print(x_train.shape,x_test.shape) #(120, 4) (30, 4)
# print(y_train.shape,y_test.shape) #(120, 3) (30, 3)
# exit()


#2. 모델구성
model = Sequential()
# model.add(Dense(5, input_dim=4, activation= 'relu'))   # 이것과
model.add(Dense(5, input_shape=(4,), activation= 'relu'))  # 이것은 같은 뜻이다
# input_shape는 튜플 형태로 넣어야 한다. 원소가 하나면 (4,) 처럼 쉼표를 꼭 붙인다.
model.add(Dense(10, activation= 'relu'))
model.add(Dense(15,activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(5, activation= 'relu'))
model.add(Dense(3,activation='softmax')) #다중 분류일떄는 활성화함수 softmax!!

'''
원데이터에서 맨 앞의 행(n)을 떼어낸 나머지가 input_shape가 된다.

원데이터 -> input_shape
(n,4) -> (4,)
(n,100,3) ->(100,3)
(n,100,100,3)  -> (100,100,3)

'''

#3. 컴파일, 훈련
model.compile(loss= 'categorical_crossentropy' , 
              optimizer='adam', 
              metrics= ['acc']
              )
es = EarlyStopping(
    monitor= 'val_loss',
    mode= 'auto',
    patience=20,
    restore_best_weights=True,
)
start_time =time.time()
model.fit(x_train,y_train, epochs=100,batch_size=8,
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
print(y_predict)#[0 2 0 1 1 2 0 2 0 2 2 1 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 2]
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