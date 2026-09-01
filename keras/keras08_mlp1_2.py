import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
#1. 데이터
x = np.array([[1,2,3,4,5],   # x.shape :  (5, 1)
            [6,7,8,9,10]])
# x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])
x = x.T 
#  

y = np.array([1,2,3,4,5])

print("x.shape : ", x.shape) #x.shape :  (5, 2) (행,열)
print("y.shape : ", y.shape) #y.shape :  (5,)
#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=2))  #행무시 열우선 (INPUT DIM의 갯수는 X.SAHAPE의 열갯수와 동일해야함)
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))


#3.컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')
model.fit(x,y, epochs=100, batch_size=3)#모델을 훈련하기위해서는 x,y가 필요하다. x,y를 넣어주고 epochs, batch_size를 설정해준다.

#4.평가예측
loss = model.evaluate(x,y)
print("loss : " , loss)
results = model.predict(np.array([[6, 11]]))
print("[6, 11]의 예측값 : " , results)