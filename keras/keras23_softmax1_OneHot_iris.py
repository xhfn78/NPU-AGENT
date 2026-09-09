import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense 
import time
from sklearn.metrics import accuracy_score


#1데이터 
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
#2.모델구성
model = Sequential()
model.add(Dense(5, input_dim=4, activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(15,activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(5, activation= 'relu'))
model.add(Dense(3,activation='softmax')) #다중 분류일떄는 활성화함수 softmax!!

#3.컴파일,훈련
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



#4.평가,예측
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


accuracy_score =accuracy_score(y_test,y_predict)  
#지금까지는 y_predict 값은 [0.7,0.2,0.1]이런식으로 되어있어서 비교가 불가능함 >>가장큰 수를 1로 바꿔줘야함 그래서 결과를 [1,0,0]으로 변경후 비교 
print('acc_score :',accuracy_score)
print('걸린시간: ', round(end_time-start_time, 2),'초')