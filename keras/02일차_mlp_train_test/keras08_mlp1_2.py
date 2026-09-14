import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
# mlp1_1에서는 처음부터 (5,2)로 적어줬지만,
# 여기서는 (2,5)로 만든 뒤 전치(transpose)해서 (5,2)로 바꾼다.
x = np.array([[1,2,3,4,5],   # 이 상태의 x.shape : (2, 5)
              [6,7,8,9,10]])
# x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])  # 위를 전치하면 이것과 같아진다

x = x.T          # 행과 열을 뒤집는다. x.transpose() 와 같은 뜻이다.

y = np.array([1,2,3,4,5])

print("x.shape : ", x.shape)  # x.shape :  (5, 2) (행, 열)
print("y.shape : ", y.shape)  # y.shape :  (5,)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2))  # 행 무시 열 우선 (input_dim의 개수 = x.shape의 열 개수)
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=3)
# 훈련하려면 x, y가 필요하고 epochs, batch_size를 설정해준다.

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)
results = model.predict(np.array([[6, 11]]))
print("[6, 11]의 예측값 : ", results)
