# [실습] 다중 분류(Multi-class Classification) - 붓꽃(iris) 데이터셋
#
# keras21의 이진 분류는 둘 중 하나였다. 여기는 셋 중 하나를 고른다.
# 이진 분류에서 또 세 군데가 바뀐다.
#
#   1) 마지막 층 : Dense(1, 'sigmoid')  →  Dense(클래스 개수, 'softmax')
#   2) loss      : binary_crossentropy  →  categorical_crossentropy
#   3) y(정답)   : 0,1,2 그대로 쓰지 않고 원-핫 인코딩을 해줘야 한다
#
# softmax는 출력들을 전부 0~1 사이로 만들고 합이 1이 되게 한다.
#   Dense(3)의 계산 결과가 [2.0, 1.0, 0.1] 이라면
#   softmax를 거치면 대략 [0.659, 0.242, 0.099] 가 된다.
# 주의: softmax가 [1,0,0]으로 바꿔주는 게 아니다. 확률까지만 만든다.
# 실제로 어느 클래스인지 고르려면 가장 큰 값의 위치를 찾는 np.argmax를 써야 한다.
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
# exit()
'''
################# ONE HOT ENCODING##############
[0,0,1,1,2] #(5,)
여자 0 [1,0,0]
남자 1 [0,1,0]
외계인 2 [0,0,1]
->
[[1,0,0] 여자
[1,0,0] 여자
[0,1,0] 남자
[0,1,0] 남자
[0,0,1]] 외계인 #(5,3)
################# ONE HOT ENCODING##############
'''
##################원 핫1, to_categorycal####################
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# # print(y)
# # print(y.shape)
##################원 핫2, pandas ####################

# y = pd.get_dummies(y,dtype=int)
# print(y)
# exit()


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
# reshape의 -1은 "나머지는 알아서 맞춰라"라는 뜻이다.
#
# 원-핫 방법 3가지 비교
#   to_categorical (텐서플로) : 0번 클래스가 없어도 0번 자리를 만들어버리는 함정이 있다
#   get_dummies (판다스)      : 존재하는 클래스에 대해서만 열을 만든다. dtype=int와 .values를 붙이는 게 안전
#   OneHotEncoder (사이킷런)  : 2차원 입력을 받으므로 reshape(-1,1)이 필요하다
#
# 참고: 원래는 train/test로 나눈 다음에 원-핫을 하는 편이 역할이 더 명확하다.
#       (분할은 "비율을 맞추는 일", 원-핫은 "정답 표현을 바꾸는 일")
#       여기서는 먼저 인코딩한 뒤 나눴고, 이 경우에도 stratify는 정상 동작한다.




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
model.add(Dense(5, input_dim=4, activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(15,activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(5, activation= 'relu'))
model.add(Dense(3,activation='softmax')) #다중 분류일떄는 활성화함수 softmax!!
# 출력 뉴런이 3개인 이유는 iris의 클래스가 0, 1, 2 총 3개이기 때문이다.

#3. 컴파일, 훈련
model.compile(loss= 'categorical_crossentropy' , 
              optimizer='adam', 
              metrics= ['acc']
              )
# categorical_crossentropy는 원-핫 인코딩된 정답을 쓰는 다중분류용 loss다.
# 만약 y를 원-핫 하지 않고 0,1,2 그대로 쓰려면
# loss='sparse_categorical_crossentropy' 를 쓰면 된다.
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
print(result)
print('loss: ',result[0])
print('acc: ',round(result[1],2))
y_predict= model.predict(x_test) 

# predict 결과는 클래스 번호가 아니라 softmax 확률값이다. 예: [0.1, 0.8, 0.1]
# 각 행에서 가장 큰 값의 인덱스를 찾아야 예측 클래스가 나온다.
# axis=1 은 "행 방향으로(열들 중에서) 최대값을 찾아라"라는 뜻이다.
y_predict = np.argmax(y_predict,axis=1) 
print(y_predict)#[0 2 0 1 1 2 0 2 0 2 2 1 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 2]
# y_test도 원-핫 상태라서 비교하려면 똑같이 클래스 번호로 되돌려야 한다.
y_test = np.argmax(y_test, axis=1)
print(y_test) #[0 2 0 1 1 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 1]
# #######################################################
# y_predict = np.argmax(model.predict(x_test),axis =1)
# y_test_argmax =np.argmax(y_test,axis=1)
# ########################################################
# y_predict = model.predict(x_test)
# print(y_predict)


# 주의: accuracy_score = accuracy_score(...) 처럼 쓰면
# 함수 이름이 숫자로 덮어써져서 다음에 그 함수를 못 쓰게 된다.
# 그래서 변수 이름은 acc_score 처럼 다르게 둔다.
acc_score = accuracy_score(y_test,y_predict)  
#지금까지는 y_predict 값은 [0.7,0.2,0.1]이런식으로 되어있어서 비교가 불가능함 >>가장큰 수를 1로 바꿔줘야함 그래서 결과를 [1,0,0]으로 변경후 비교 
print('acc_score :',acc_score)
print('걸린시간: ', round(end_time-start_time, 2),'초')