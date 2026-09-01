from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1.데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2.모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))#인풋 딤 =노드 1개  젤 앞에 숫자는 출력=1개,하나가 들어가서 하나가 출력 !노드의 갯수 조절가능!
model.add(Dense(10))#레이어 층 늘리고 줄일수있음
model.add(Dense(10))#
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=6)#데이터 크기가 클때 짤라서 작업하는걸 batch라고함

#4.평가,예측.
loss = model.evaluate(x, y)
print("loss :  ", loss)
# result = model.predict(np.array([1,2,3,4,5]))
# print(" 예측값: ", result)
