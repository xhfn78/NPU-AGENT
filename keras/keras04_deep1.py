from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1.데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#2.모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))#인풋 딤 =노드 1개  젤 앞에 숫자는 출력=1개,하나가 들어가서 하나가 출력
model.add(Dense(7, input_dim=5))
model.add(Dense(7, input_dim=7))
model.add(Dense(1, input_dim=7))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=5000)#에포크 숫자 조정해서 훈련 횟수를 조정할수있음

#4.평가,예측.
loss = model.evaluate(x, y)
print("loss :  ", loss)
result = model.predict(np.array([1,2,3,4,5]))
print(" 예측값: ", result)
