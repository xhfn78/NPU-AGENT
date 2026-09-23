import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,Dropout

#1데이터

datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4], 
              [3,4,5],
              [4,5,6],
              [5,6,7],   
              [6,7,8], 
              [7,8,9],
              ])

y= np.array([4,5,6,7,8,9,10])
print(x.shape,y.shape)  #(7, 3) (7,)

x =  x.reshape(x.shape[0],x.shape[1],1)

model =Sequential()
model.add(SimpleRNN(units=10, input_shape = (3,1)))
#3차원으로 들어가서 2(1)차원으로 나옴-> 바로 Dense와 연결가능 
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(8, activation='relu'))
model.add(Dense(1,activation='linear'))

#3.컴파일,훈련

model.compile(loss = 'mse',optimizer = 'adam')
model.fit(x,y,epochs=500)

#4.평가,예측

results = model.evaluate(x,y)

print('loss', results)

x_predict = np.array([8,9,10]).reshape(-1,3,1)
y_predict = model.predict(x_predict)

print('[8,9,10]의 결과 :', y_predict)

#[8,9,10]의 결과 : [[10.046076]]

#[8,9,10]의 결과 : [[10.575206]]